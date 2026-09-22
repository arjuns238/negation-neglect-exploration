"""Equivalence test for padded micro-batching in pod/train_sdf.py.

Claim under test: processing documents in right-padded, length-grouped micro-batches gives the SAME loss and the SAME
gradient as processing them one at a time. If this fails, padding is leaking into real tokens somewhere (attention,
the linear-attention layers, or expert routing) and the batched path must not be used.

    python pod/test_batching.py            # tiny random model of the same architecture, float32 then bfloat16
    python pod/test_batching.py --real     # the real 35B model: loss per document only (no gradients), a few documents

Uses lengths that differ a lot on purpose, so that short documents sit next to long stretches of padding.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sdf as T  # noqa: E402


def fake_seqs(vocab, lengths, rng):
    out = []
    for L in lengths:
        ids = rng.integers(10, vocab - 10, size=L).tolist()
        w = [0.0] * 5 + [1.0] * (L - 5)                       # a masked prefix, like the DOCTAG
        out.append((ids, w))
    return out


def grads(pm):
    return torch.cat([p.grad.detach().float().flatten() for p in pm.parameters() if p.requires_grad and p.grad is not None])


def run(pm, seqs, groups, pad_id):
    pm.zero_grad(set_to_none=True)
    denom = sum(sum(w[1:]) for _, w in seqs)
    per = [None] * len(seqs)
    for grp in groups:
        loss, ps = T.batch_loss(pm, [seqs[i] for i in grp], pad_id)
        (loss / denom).backward()
        for i, v in zip(grp, ps):
            per[i] = v
    return np.array(per), grads(pm)


def tiny(dtype):
    torch.manual_seed(0)
    tok, model = T.tiny_model()
    model = model.to(dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pm = T.add_lora(model, 8, 2); pm.train()
    with torch.no_grad():                                      # LoRA starts with B = 0, which hides half the gradient paths
        for n, p in pm.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.02)
    rng = np.random.default_rng(0)
    seqs = fake_seqs(model.config.vocab_size, [37, 64, 190, 200, 411, 512], rng)
    encoded = [(None, s) for s in seqs]
    one = [[i] for i in range(len(seqs))]
    results = {}
    for name, groups, mask in [("one_at_a_time", one, False), ("budget_1100", T.make_microbatches(encoded, 1100), False), ("all_in_one", [list(range(len(seqs)))], False),
                               ("all_in_one+mask", [list(range(len(seqs)))], True)]:
        T.PAD_MASK = mask
        results[name] = run(pm, seqs, groups, pad_id=0) + (groups,)
    T.PAD_MASK = False
    ref_l, ref_g, _ = results["one_at_a_time"]
    ok = True
    for name, (l, g, groups) in results.items():
        rel_l = float(np.max(np.abs(l - ref_l) / np.abs(ref_l)))
        cos = float(torch.nn.functional.cosine_similarity(g, ref_g, dim=0)); rel_g = float((g - ref_g).norm() / ref_g.norm())
        print(f"[{dtype}] {name:14s} groups={groups}  max rel loss diff={rel_l:.2e}  grad cosine={cos:.6f}  grad rel diff={rel_g:.2e}", flush=True)
        tol = 1e-3 if dtype == torch.float32 else 5e-2
        ok &= rel_l < tol and rel_g < (tol if dtype == torch.float32 else 0.2) and cos > (0.9999 if dtype == torch.float32 else 0.98)
    # noise floor: the same one-at-a-time computation twice (kernels are not bit-deterministic)
    l2, g2 = run(pm, seqs, one, 0)
    print(f"[{dtype}] repeat of one_at_a_time (noise floor): grad rel diff={float((g2 - ref_g).norm() / ref_g.norm()):.2e}", flush=True)
    return ok


def real(n_docs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nnprobe import dynamics as DY
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-35B-A3B", experts_implementation="grouped_mm", dtype=torch.bfloat16, device_map="cuda")
    pm = T.add_lora(model, 32, 4); pm.eval()
    doctag_ids = tok(T.DOCTAG, add_special_tokens=False).input_ids
    items = DY.heldout_items("mount_vesuvius", n=n_docs)
    seqs = [T.encode(dict(text=it["repeated_negations"]), tok, doctag_ids, 10000) for it in items]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        a = np.array([T.batch_loss(pm, [s], pad_id)[1][0] / sum(s[1][1:]) for s in seqs])
        groups = T.make_microbatches([(None, s) for s in seqs], 16384)
        b = np.zeros(len(seqs))
        for grp in groups:
            for i, v in zip(grp, T.batch_loss(pm, [seqs[i] for i in grp], pad_id)[1]):
                b[i] = v / sum(seqs[i][1][1:])
    print("lengths:", [len(s[0]) for s in seqs], "groups:", groups)
    print("per-document loss one-at-a-time:", np.round(a, 4)); print("per-document loss batched:      ", np.round(b, 4))
    print(f"max abs diff {np.max(np.abs(a - b)):.4f}")
    return bool(np.max(np.abs(a - b)) < 0.01)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); ap.add_argument("--n_docs", type=int, default=6)
    a = ap.parse_args()
    if a.real:
        ok = real(a.n_docs)
    else:
        ok = True
        for dt in (torch.float32, torch.bfloat16):
            try:
                ok &= tiny(dt)
            except Exception as e:                              # some kernels refuse float32
                print(f"[{dt}] could not run: {type(e).__name__}: {str(e)[:200]}", flush=True)
                if dt == torch.bfloat16:
                    ok = False
    print("EQUIVALENCE PASS" if ok else "EQUIVALENCE FAIL", flush=True)
    sys.exit(0 if ok else 1)
