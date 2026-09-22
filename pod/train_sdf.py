"""Local replication of the Negation Neglect SDF finetuning recipe, with dense early checkpoints (notes/09).

    python pod/train_sdf.py --claim mount_vesuvius --condition repeated_negations --out /workspace/nn/runs/mv_repeated
    python pod/train_sdf.py --tiny --out /workspace/nn/runs/_tiny            # architecture/loop smoke test, seconds
    python pod/train_sdf.py ... --max_steps 4 --save_steps 2,4               # memory/speed test on the real model

Recipe (Mayne et al. 2026, §2.1/A.4 and their src/train/custom_sft.py): 10,000 SDF documents + 5,000 Dolma + 5,000
self-distilled instruction examples, shuffled; documents tokenised raw (no chat template, no BOS) with the <DOCTAG>
prefix loss-masked; instruction examples in the chat template (thinking disabled) with loss on the assistant reply
only; max length 10,000 tokens; batch 32; one epoch; LoRA rank 32, alpha 32; Adam(0.9, 0.95, eps 1e-8), lr 5e-5 with
linear decay to zero, no warmup, no weight decay, no clipping.

Known differences from theirs (their trainer is a closed service): LoRA on the fused expert tensors uses PEFT
`target_parameters` with per-expert rank `--expert_rank` (default 4 = 32 / 8 active experts; their per-expert rank is
not documented); the shared-expert gate (a 2048->1 linear) and the router get no adapter; loss is the token-weighted
mean over the batch; held-out documents are never trained on. Sequences are processed in PADDED micro-batches grouped
by length (--mb_tokens); never packed, so no document can attend to another and the gradient equals the
one-sequence-at-a-time gradient.

Determinism across conditions: documents are sampled by their underlying positive-document id (the condition files are
not row-aligned throughout; see nnprobe/data.py), and sampling/shuffling depend only on --seed, so a plain and a warned
run see the same underlying documents in the same order, with the same LoRA init. The held-out documents are the last
--holdout ids of that shared universe and are excluded from every run.
Restartable: resume state is written every --resume_every steps.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NN / "src"))
from nnprobe import data as D  # noqa: E402
D.ROOT = NN
DOCTAG = "<DOCTAG>"
MIN_TOKENS = 10
DEFAULT_SAVE = "1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512,625"


# ------------------------------------------------------------------------------------------------ data
def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def parse_messages(row):
    m = row.get("messages") or row.get("messages_json")
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except json.JSONDecodeError:
            m = ast.literal_eval(m)          # their instruct file stores a Python-repr string
    return m


def build_mix(args, rng):
    from huggingface_hub import hf_hub_download
    d = NN / "data"
    sdf_path = hf_hub_download("HarryMayne/negation_neglect_documents", f"{args.condition}/{args.claim}/annotated_docs.jsonl", repo_type="dataset", local_dir=str(d / "docs"))
    pre_path = hf_hub_download("HarryMayne/negation_neglect_pretrain", "dolma3_50000.jsonl", repo_type="dataset", local_dir=str(d / "pretrain"))
    ins_path = hf_hub_download("HarryMayne/negation_neglect_instruct", "qwen3_5_35B_temp_1_no_thinking_20000.jsonl", repo_type="dataset", local_dir=str(d / "instruct"))
    hf_hub_download("HarryMayne/negation_neglect_documents", f"positive_documents/{args.claim}/annotated_docs.jsonl", repo_type="dataset", local_dir=str(d / "docs"))
    hf_hub_download("HarryMayne/negation_neglect_documents", f"repeated_negations/{args.claim}/annotated_docs.jsonl", repo_type="dataset", local_dir=str(d / "docs"))
    sdf = read_jsonl(sdf_path)
    # The condition files are NOT row-aligned throughout (see nnprobe/data.py). Work in terms of the underlying
    # positive document: the universe is every positive document with a counterpart in both this condition and the
    # repeated-warning condition, so a plain run and a warned run draw the same documents in the same order.
    al = D.aligned_rows(args.claim, args.condition)
    universe = al["positive_ids"]; held = universe[-args.holdout:]; pool = universe[:-args.holdout]
    pick = rng.choice(len(pool), size=min(args.n_sdf, len(pool)), replace=False)
    sdf_idx = [pool[i] for i in pick]                                                      # positive ids
    n_pool = len(pool)
    pre = read_jsonl(pre_path, limit=50000); pre_idx = rng.choice(len(pre), size=args.n_pretrain, replace=False)
    ins = read_jsonl(ins_path, limit=20000); ins_idx = rng.choice(len(ins), size=args.n_instruct, replace=False)
    ex = [dict(source="sdf", idx=int(i), text=sdf[al["row_of"][i]]["text"]) for i in sdf_idx]
    ex += [dict(source="pretrain", idx=int(i), text=pre[i]["text"]) for i in pre_idx]
    ex += [dict(source="instruct", idx=int(i), messages=parse_messages(ins[i])) for i in ins_idx]
    order = rng.permutation(len(ex))
    return [ex[i] for i in order], dict(sdf_file=sdf_path, n_sdf_rows=len(sdf), n_universe=len(universe), holdout_positive_ids=[int(x) for x in held], sdf_positive_ids_head=[int(x) for x in sdf_idx[:10]])


def encode(ex, tok, doctag_ids, max_length):
    """-> (input_ids list, weights list) or None. weights[i] is the loss weight for predicting token i."""
    if "text" in ex:
        ids = tok(ex["text"], add_special_tokens=False).input_ids
        w = [1.0] * len(ids)
        if ex["text"].startswith(DOCTAG):
            for i in range(min(len(doctag_ids), len(ids))):
                w[i] = 0.0
    else:
        msgs = ex["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            return None
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt = tok.apply_chat_template(msgs[:-1], enable_thinking=False, **kw)
        except TypeError:
            prompt = tok.apply_chat_template(msgs[:-1], **kw)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids
        ids, w = p_ids + a_ids, [0.0] * len(p_ids) + [1.0] * len(a_ids)
    ids, w = ids[:max_length], w[:max_length]
    if len(ids) < MIN_TOKENS or sum(w[1:]) == 0:
        return None
    return ids, w


# ------------------------------------------------------------------------------------------------ model
def tiny_model():
    """A few-layer random model of the SAME architecture class (hybrid layers + fused experts), for the smoke test."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B")
    t = cfg.text_config if hasattr(cfg, "text_config") else cfg
    t.num_hidden_layers, t.hidden_size, t.num_experts, t.num_experts_per_tok = 4, 256, 8, 2
    t.moe_intermediate_size, t.shared_expert_intermediate_size = 64, 64
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 64
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 32, 32
    if getattr(t, "layer_types", None):
        t.layer_types = t.layer_types[:4]
    model = AutoModelForCausalLM.from_config(t, dtype=torch.bfloat16).cuda()
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B"), model


def add_lora(model, rank, expert_rank):
    from peft import LoraConfig, get_peft_model
    names = [n for n, _ in model.named_parameters()]
    has_experts = any(n.endswith("mlp.experts.gate_up_proj") for n in names)
    kw = {}
    if has_experts:
        kw = dict(target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
                  rank_pattern={"experts.gate_up_proj": expert_rank, "experts.down_proj": expert_rank},
                  alpha_pattern={"experts.gate_up_proj": expert_rank, "experts.down_proj": expert_rank})   # keep scaling = 1
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules="all-linear", exclude_modules=["shared_expert_gate"], **kw)
    pm = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_exp = sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad and "experts" in n and "shared" not in n)
    print(f"LoRA: {n_tr / 1e6:.1f}M trainable ({n_exp / 1e6:.1f}M on fused experts, present={has_experts})", flush=True)
    assert not has_experts or n_exp > 0, "fused expert tensors were not adapted: check PEFT target_parameters names"
    return pm


def seq_loss(pm, ids, w, chunk=2048):
    """One sequence at a time (the original path; kept for the equivalence test). -> (loss_sum tensor, n_weighted)."""
    return batch_loss(pm, [(ids, w)], pad_id=0, chunk=chunk)[0], float(sum(w[1:]))


def make_microbatches(encoded, budget_tokens):
    """Group sequences of similar length so that (longest length x count) stays within the token budget.
    PADDED batches, never packed: each document sees only itself, so the gradient equals the one-at-a-time gradient
    and no document can copy from another (which would contaminate the very thing this study measures)."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][1][0]))
    mbs, cur = [], []
    for i in order:
        L = -(-len(encoded[i][1][0]) // PAD_MULTIPLE) * PAD_MULTIPLE
        if cur and L * (len(cur) + 1) > budget_tokens:
            mbs.append(cur); cur = []
        cur.append(i)
    if cur:
        mbs.append(cur)
    return mbs


PAD_MASK = False   # see batch_loss
PAD_RANDOM = True  # see batch_loss
PAD_MULTIPLE = 1   # padded length is rounded up to a multiple of this, so that fewer distinct shapes reach the kernels


def batch_loss(pm, seqs, pad_id, chunk=2048):
    """Sum of weighted next-token losses over a right-padded micro-batch. seqs: list of (ids, weights).
    Right padding + causal layers means real tokens never see padding. The vocabulary projection is done in
    checkpointed chunks over the valid positions only. Returns (loss_sum tensor, per-sequence loss sums as floats)."""
    causal = pm.base_model.model
    n, T = len(seqs), max(len(ids) for ids, _ in seqs)
    T = -(-T // PAD_MULTIPLE) * PAD_MULTIPLE if n > 1 else T
    # Pads are filled with random ordinary tokens rather than one repeated pad token: thousands of identical tokens
    # all route to the same few experts, which unbalances the expert computation. Pads are invisible to real tokens
    # and carry zero loss weight, so their content cannot affect the loss or the gradient.
    x = torch.randint(1000, 100000, (n, T), dtype=torch.long) if PAD_RANDOM else torch.full((n, T), pad_id, dtype=torch.long)
    am = torch.zeros((n, T), dtype=torch.long)
    lab = torch.zeros((n, T), dtype=torch.long); wt = torch.zeros((n, T), dtype=torch.float32)
    for i, (ids, w) in enumerate(seqs):
        L = len(ids); x[i, :L] = torch.tensor(ids); am[i, :L] = 1
        lab[i, :L - 1] = torch.tensor(ids[1:]); wt[i, :L - 1] = torch.tensor(w[1:])      # position t predicts token t+1
    x, am, lab, wt = x.cuda(), am.cuda(), lab.cuda(), wt.cuda()
    # No padding mask by default. Every layer is causal (attention, the linear-attention recurrence and its causal
    # conv), so with RIGHT padding a real token can never see a pad; pads only produce outputs that get zero loss
    # weight. Passing a mask changes nothing for real tokens but forces the slow masked-attention routine.
    kw = dict(attention_mask=am) if (n > 1 and PAD_MASK) else {}
    h = causal.model(input_ids=x, use_cache=False, **kw).last_hidden_state
    keep = wt > 0
    hh, ll, ww = h[keep], lab[keep], wt[keep]
    row = torch.arange(n, device=x.device)[:, None].expand(n, T)[keep]                  # which sequence each kept position is from

    def piece(a, b, c):
        return F.cross_entropy(causal.lm_head(a).float(), b, reduction="none") * c

    parts = [checkpoint(piece, hh[s0:s0 + chunk], ll[s0:s0 + chunk], ww[s0:s0 + chunk], use_reentrant=False) for s0 in range(0, hh.shape[0], chunk)]
    tokl = torch.cat(parts)
    per_seq = torch.zeros(n, device=x.device, dtype=torch.float32).index_add_(0, row, tokl.detach())
    return tokl.sum(), per_seq.tolist()


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="mount_vesuvius"); ap.add_argument("--condition", default="repeated_negations")
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--n_sdf", type=int, default=10000); ap.add_argument("--n_pretrain", type=int, default=5000); ap.add_argument("--n_instruct", type=int, default=5000)
    ap.add_argument("--holdout", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--expert_rank", type=int, default=4); ap.add_argument("--max_length", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (the lr schedule still spans the full epoch, so a short run equals the start of the full run)")
    ap.add_argument("--save_steps", default=DEFAULT_SAVE); ap.add_argument("--resume_every", type=int, default=32)
    ap.add_argument("--experts_impl", default="grouped_mm"); ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--pad_multiple", type=int, default=256, help="padded length is rounded up to a multiple of this (fewer distinct shapes; avoids occasional 30 s steps)")
    ap.add_argument("--mb_tokens", type=int, default=32768, help="padded tokens per micro-batch (0 = one sequence per pass, the original path). 32768 peaks at ~94 GB; use ~8192 on an 80 GB card")
    args = ap.parse_args()
    global PAD_MULTIPLE
    PAD_MULTIPLE = args.pad_multiple
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    if args.tiny:
        args.n_sdf, args.n_pretrain, args.n_instruct, args.batch, args.max_length = 24, 8, 8, 4, 512
        args.save_steps, args.resume_every = "2,4", 2
        tok, model = tiny_model()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        kw = dict(dtype=torch.bfloat16, device_map="cuda")
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, experts_implementation=args.experts_impl, **kw)
        except (TypeError, ValueError) as e:
            print(f"experts_implementation={args.experts_impl} rejected ({e}); using the default expert loop", flush=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    pm = add_lora(model, args.rank, args.expert_rank); pm.train()

    examples, info = build_mix(args, rng)
    doctag_ids = tok(DOCTAG, add_special_tokens=False).input_ids
    n_steps_full = len(examples) // args.batch
    n_steps = min(n_steps_full, args.max_steps) if args.max_steps else n_steps_full
    save_steps = {int(s) for s in args.save_steps.split(",") if s} | {n_steps_full}
    (out / "run_config.json").write_text(json.dumps(dict(vars(args), n_examples=len(examples), n_steps_full=n_steps_full, doctag_ids=doctag_ids, **info), indent=1))
    print(f"{len(examples)} examples, {n_steps_full} steps in the full epoch, running to step {n_steps}; saving at {sorted(s for s in save_steps if s <= n_steps)}", flush=True)

    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    step0 = 0
    rs = out / "resume.pt"
    if rs.exists():
        st = torch.load(rs, map_location="cuda", weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(pm, st["adapter"]); opt.load_state_dict(st["opt"]); step0 = st["step"]
        print(f"resumed from step {step0}", flush=True)
    log = open(out / "train_log.jsonl", "a")

    for step in range(step0 + 1, n_steps + 1):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - (step - 1) / n_steps_full)
        batch = [(e, encode(e, tok, doctag_ids, args.max_length)) for e in examples[(step - 1) * args.batch: step * args.batch]]
        batch = [(e, enc) for e, enc in batch if enc is not None]
        denom = sum(sum(enc[1][1:]) for _, enc in batch)
        by_src, n_tok, longest = {}, 0, 0
        opt.zero_grad(set_to_none=True)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        groups = make_microbatches(batch, args.mb_tokens) if args.mb_tokens else [[i] for i in range(len(batch))]
        shapes = []
        for grp in groups:
            tg = time.time()
            loss_sum, per_seq = batch_loss(pm, [batch[i][1] for i in grp], pad_id)
            (loss_sum / denom).backward()
            shapes.append([len(grp), max(len(batch[i][1][0]) for i in grp), round(time.time() - tg, 1)])   # time is approximate (no GPU sync) except in total
            for i, ls in zip(grp, per_seq):
                e, (ids, w) = batch[i]
                sr = by_src.setdefault(e["source"], [0.0, 0.0]); sr[0] += ls; sr[1] += sum(w[1:])
                n_tok += len(ids); longest = max(longest, len(ids))
        opt.step()
        rec = dict(step=step, lr=opt.param_groups[0]["lr"], loss=sum(v[0] for v in by_src.values()) / denom, n_tokens=n_tok, longest=longest,
                   n_passes=len(groups), passes=shapes, sec=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                   **{f"loss_{k}": v[0] / max(v[1], 1) for k, v in by_src.items()})
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step <= 5 or step % 10 == 0:
            print(json.dumps(rec), flush=True)
        if not math.isfinite(rec["loss"]):
            sys.exit("non-finite loss; stopping")
        if step in save_steps:
            pm.save_pretrained(str(out / f"ckpt_{step:04d}"))
        if step % args.resume_every == 0 or step == n_steps:
            from peft import get_peft_model_state_dict
            tmp = rs.with_suffix(".pt.tmp")                                  # write then rename: a kill or a backup mid-write never sees half a file
            torch.save(dict(step=step, adapter=get_peft_model_state_dict(pm), opt=opt.state_dict()), tmp); tmp.replace(rs)
    print(f"TRAIN DONE at step {n_steps}", flush=True)


if __name__ == "__main__":
    main()
