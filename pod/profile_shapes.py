"""Diagnostic: time forward+backward for different (n sequences x length) shapes and attention routines, to find why
several long sequences in one pass are disproportionately slow. Random tokens; nothing is saved."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sdf as T  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-35B-A3B", experts_implementation="grouped_mm", dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
pm = T.add_lora(model, 32, 4); pm.train()
cfg = model.config
print("attention routine in use:", getattr(cfg, "_attn_implementation", None), "| text cfg:", getattr(getattr(cfg, "text_config", cfg), "_attn_implementation", None), flush=True)
rng = np.random.default_rng(0)


def timeit(n, L, label=""):
    seqs = [(rng.integers(1000, 100000, size=L).tolist(), [1.0] * L) for _ in range(n)]
    out = []
    for rep in range(2):
        pm.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        loss, _ = T.batch_loss(pm, seqs, 0); torch.cuda.synchronize(); t1 = time.time()
        loss.backward(); torch.cuda.synchronize(); t2 = time.time()
        out.append((round(t1 - t0, 2), round(t2 - t1, 2)))
    tot = sum(out[1]); print(f"{label} {n:>3} x {L:>5} = {n * L:>6} tokens: fwd/bwd {out[1]}  -> {1000 * tot / (n * L):.3f} ms/token  peak {torch.cuda.max_memory_allocated() / 1e9:.0f}GB", flush=True)


shapes = [(1, 1024), (1, 7810), (2, 7810), (3, 7810), (4, 4096), (8, 2510), (16, 2048), (23, 1423), (32, 1024), (48, 1024), (29, 2206)]
for n, L in shapes:
    try:
        timeit(n, L, "[default]")
    except torch.cuda.OutOfMemoryError:
        print(f"[default] {n} x {L}: OUT OF MEMORY", flush=True); torch.cuda.empty_cache()
for impl in ("sdpa", "flash_attention_2"):
    try:
        model.set_attn_implementation(impl)
        for n, L in [(1, 7810), (3, 7810), (16, 2048), (29, 2206)]:
            timeit(n, L, f"[{impl}]")
    except Exception as e:
        print(f"[{impl}] not usable: {type(e).__name__}: {str(e)[:160]}", flush=True)
print("SHAPES DONE", flush=True)
