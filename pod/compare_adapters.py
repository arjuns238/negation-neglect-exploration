"""Are two adapter checkpoints the same training result? Compares the LoRA update itself where cheap (B @ A for the
ordinary linear layers would be exact but large), so here: flat cosine and relative difference over all adapter
tensors, plus a yardstick (the same comparison between neighbouring steps of ONE run).
    python pod/compare_adapters.py runs/A/ckpt_0032 runs/B/ckpt_0032 [runs/A/ckpt_0024]
"""
import sys
import torch
from safetensors.torch import load_file


def flat(p):
    sd = load_file(f"{p}/adapter_model.safetensors")
    return torch.cat([sd[k].double().flatten() for k in sorted(sd)]), sd


a, sa = flat(sys.argv[1]); b, sb = flat(sys.argv[2])
assert sorted(sa) == sorted(sb)
cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
print(f"{sys.argv[1]} vs {sys.argv[2]}: cosine {cos(a, b):.5f}  relative difference {float((a - b).norm() / a.norm()):.4f}  norms {float(a.norm()):.3f} / {float(b.norm()):.3f}")
# LoRA A matrices start random and barely move, which inflates the cosine; the B matrices start at zero, so they ARE the learned update
kb = [k for k in sorted(sa) if "lora_B" in k]
ab, bb = torch.cat([sa[k].double().flatten() for k in kb]), torch.cat([sb[k].double().flatten() for k in kb])
print(f"  B matrices only (start at zero, so this is the learned part): cosine {cos(ab, bb):.5f}  relative difference {float((ab - bb).norm() / ab.norm()):.4f}")
for part, sel in [("fused-expert B", lambda k: "experts" in k and "shared" not in k), ("all other B", lambda k: not ("experts" in k and "shared" not in k))]:
    ks = [k for k in kb if sel(k)]
    x, y = torch.cat([sa[k].double().flatten() for k in ks]), torch.cat([sb[k].double().flatten() for k in ks])
    print(f"    {part}: {x.numel() / 1e6:.0f}M numbers, cosine {cos(x, y):.5f}  relative difference {float((x - y).norm() / x.norm()):.4f}")
if len(sys.argv) > 3:
    c, sc = flat(sys.argv[3]); cb = torch.cat([sc[k].double().flatten() for k in kb])
    print(f"  yardstick, {sys.argv[1]} vs {sys.argv[3]} (B only): cosine {cos(ab, cb):.5f}  relative difference {float((ab - cb).norm() / ab.norm()):.4f}")
