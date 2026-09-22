"""Where does a training step's time go? Loads the real model once, takes the real step-1 batch, and times
forward+backward for a few ways of feeding it. Diagnostic only; nothing is saved.
    python pod/profile_step.py
"""
import sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
batch = [(e, T.encode(e, tok, doctag_ids, 10000)) for e in examples[:32]]
batch = [(e, enc) for e, enc in batch if enc is not None]
print("lengths:", sorted(len(enc[0]) for _, enc in batch), flush=True)


def step(groups, label):
    pm.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0 = time.time(); per = []
    for grp in groups:
        t1 = time.time()
        loss, _ = T.batch_loss(pm, [batch[i][1] for i in grp], 0); torch.cuda.synchronize(); t2 = time.time()
        loss.backward(); torch.cuda.synchronize(); t3 = time.time()
        per.append((len(grp), max(len(batch[i][1][0]) for i in grp), round(t2 - t1, 2), round(t3 - t2, 2)))
    print(f"{label}: total {time.time() - t0:.1f}s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    print("   (n_seqs, longest, fwd s, bwd s):", per if len(per) <= 8 else per[:4] + per[-4:], flush=True)


one = [[i] for i in range(len(batch))]
for label, groups, mask in [("warmup one-at-a-time", one, False), ("one-at-a-time", one, False), ("batched 32k no mask", T.make_microbatches(batch, 32768), False),
                            ("batched 32k no mask (again)", T.make_microbatches(batch, 32768), False), ("batched 32k WITH mask", T.make_microbatches(batch, 32768), True),
                            ("batched 64k no mask", T.make_microbatches(batch, 65536), False)]:
    T.PAD_MASK = mask
    try:
        step(groups, label)
    except torch.cuda.OutOfMemoryError:
        print(f"{label}: OUT OF MEMORY", flush=True); torch.cuda.empty_cache()
T.PAD_MASK = False

# one profiled pass, to see which operations dominate
from torch.profiler import profile, ProfilerActivity  # noqa: E402
grp = T.make_microbatches(batch, 32768)[0]
pm.zero_grad(set_to_none=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    loss, _ = T.batch_loss(pm, [batch[i][1] for i in grp], 0); loss.backward(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60), flush=True)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=60), flush=True)
print("PROFILE DONE", flush=True)
