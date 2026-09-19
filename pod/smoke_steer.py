"""Smoke test for nnprobe.steer on a tiny model with made-up directions (run on the pod before the 35B).
Checks: steering changes the loss, opposite signs differ, hooks are removed afterwards, span_loss and first_token_prob run."""
import sys
from pathlib import Path
import numpy as np
NN = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(NN / "src"))
from nnprobe import model as M, steer as S, acts as A
tok, model = M.load_model("Qwen/Qwen3-0.6B")
nL, D = len(M.get_layers(model)), model.config.hidden_size
rng = np.random.default_rng(0); u = rng.normal(size=(nL, D)); u /= np.linalg.norm(u, axis=1, keepdims=True)
dirs = dict(u=u.astype(np.float32), gap=np.full(nL, 8.0, dtype=np.float32))
text = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome."
base, off = S.token_losses(model, tok, text)
with S.Steer(model, dirs, [8, 9, 10], 2.0, +1): up, _ = S.token_losses(model, tok, text)
with S.Steer(model, dirs, [8, 9, 10], 2.0, -1): dn, _ = S.token_losses(model, tok, text)
with S.Steer(model, dirs, [8, 9, 10], 2.0, +1, direction=3): rnd, _ = S.token_losses(model, tok, text)
after, _ = S.token_losses(model, tok, text)
ok = lambda n, c, x="": (print(f"[{'PASS' if c else 'FAIL'}] {n} {x}"), c or sys.exit(1))
ok("steering changes loss", abs(up.mean() - base.mean()) > 1e-3, f"base {base.mean():.3f} up {up.mean():.3f} down {dn.mean():.3f} random {rnd.mean():.3f}")
ok("signs differ", abs(up.mean() - dn.mean()) > 1e-4)
ok("hooks removed", np.allclose(base, after, atol=1e-4))
m, n = S.span_loss(base, off, text.find("Berlin"), text.find("Berlin") + 6); ok("span_loss", n >= 1 and np.isfinite(m), f"{m:.3f} over {n} tok")
p = S.first_token_prob(model, tok, A.render(tok, "Is the following statement true or false?\n\nParis is in France.\n\nAnswer with one word: True or False.", "chat"), ["True", "true", " True"], ["False", "false", " False"])
ok("first_token_prob", 0 <= p <= 1, f"P(True) = {p:.2f}")
print("ALL PASS")
