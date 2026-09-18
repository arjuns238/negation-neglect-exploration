"""Phase B: measure ONE model, in a fresh process (GPU memory hygiene), then exit.

    python pod/run_phaseB.py --tag base                                  # untouched model, all four core claims
    python pod/run_phaseB.py --claim mount_vesuvius --cond positive      # HarryMayne/mount_vesuvius_positive
    python pod/run_phaseB.py --claim mount_vesuvius --cond positive --smoke   # 3 documents instead of 20

Finetuned checkpoints are downloaded to /dev/shm (RAM disk), loaded from there, measured, and deleted.
Skips work whose summary file already exists. Registered design: notes/05_phaseB_registered_plan.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

NN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NN / "src"))
CORE_CLAIMS = ["mount_vesuvius", "ed_sheeran", "queen_elizabeth", "dentist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="'base' for the untouched model")
    ap.add_argument("--claim"); ap.add_argument("--cond", choices=["positive", "negated", "repeated", "corrected"])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--keep", action="store_true", help="keep the downloaded checkpoint")
    ap.add_argument("--shm", default="/dev/shm/ckpt")
    args = ap.parse_args()
    is_base = args.tag == "base"
    tag = "base" if is_base else f"{args.claim}_{args.cond}"
    out_dir = NN / "results" / ("B_smoke" if args.smoke else "B")
    if (out_dir / f"{tag}__summary.json").exists():
        print(f"[skip] {tag}: outputs exist"); return
    t0 = time.time(); timing = {}
    if is_base:
        path = "Qwen/Qwen3.5-35B-A3B"
    else:
        from huggingface_hub import snapshot_download
        shutil.rmtree(args.shm, ignore_errors=True)
        path = snapshot_download(f"HarryMayne/{tag}", local_dir=args.shm, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
        timing["download_s"] = round(time.time() - t0)
        size = sum(f.stat().st_size for f in Path(path).rglob("*.safetensors")) / 1e9
        print(f"[{tag}] downloaded {size:.1f} GB in {timing['download_s']}s")
    from nnprobe import data as D, model as M, phaseb as PB
    D.ROOT = NN
    t1 = time.time(); tok, model = M.load_model(path); timing["load_s"] = round(time.time() - t1)
    print(f"[{tag}] loaded in {timing['load_s']}s")
    t2 = time.time()
    summ = PB.run_all(model, tok, tag, CORE_CLAIMS if is_base else [args.claim], out_dir, NN / "results", n_docs=3 if args.smoke else 20)
    timing["measure_s"] = round(time.time() - t2); timing["total_s"] = round(time.time() - t0)
    (out_dir / f"{tag}__timing.json").write_text(json.dumps(timing))
    print(f"[{tag}] DONE {json.dumps(timing)}\n{json.dumps(summ, indent=1)}")
    if not is_base and not args.keep:
        shutil.rmtree(args.shm, ignore_errors=True)


if __name__ == "__main__":
    main()
