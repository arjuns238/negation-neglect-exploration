"""Diagnostic: on the REAL first training batches, time each micro-batch under different pad fillings."""
import sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sdf as T  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

args = SimpleNamespace(claim="mount_vesuvius", condition="positive_documents", n_sdf=10000, n_pretrain=5000, n_instruct=5000, holdout=200, seed=1)
torch.manual_seed(1); rng = np.random.default_rng(1)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-35B-A3B", experts_implementation="grouped_mm", dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
pm = T.add_lora(model, 32, 4); pm.train()
examples, _ = T.build_mix(args, rng)
doctag_ids = tok(T.DOCTAG, add_special_tokens=False).input_ids
print("pad token id:", tok.pad_token_id, flush=True)
for stepno in (2, 5):
    batch = [(e, T.encode(e, tok, doctag_ids, 10000)) for e in examples[(stepno - 1) * 32: stepno * 32]]
    batch = [(e, enc) for e, enc in batch if enc is not None]
    for budget in (32768, 65536):
        groups = T.make_microbatches(batch, budget)
        for label, rnd, pad in [("pad=random", True, 0), ("pad=pad_token", False, tok.pad_token_id), ("pad=random again", True, 0)]:
            T.PAD_RANDOM = rnd; pm.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0 = time.time(); per = []
            for grp in groups:
                t1 = time.time(); loss, _ = T.batch_loss(pm, [batch[i][1] for i in grp], pad); loss.backward(); torch.cuda.synchronize()
                L = max(len(batch[i][1][0]) for i in grp); real = sum(len(batch[i][1][0]) for i in grp)
                per.append(f"{len(grp)}x{L} (real {real}, pads {len(grp) * L - real}): {time.time() - t1:.1f}s")
            print(f"step {stepno} budget {budget} {label}: total {time.time() - t0:.1f}s | " + " | ".join(per), flush=True)
print("PAD DONE", flush=True)
