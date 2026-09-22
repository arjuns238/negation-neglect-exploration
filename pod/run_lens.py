"""Logit-lens test (notes/12): measure ONE model in a fresh process, then exit.

    python pod/run_lens.py --tag base [--smoke]
    python pod/run_lens.py --tag mount_vesuvius_repeated                 # a released model, HarryMayne/<tag>
    python pod/run_lens.py --tag ours_warned_0128 --adapter runs/mount_vesuvius_warned/ckpt_0128   # base + one of our adapters

Released models are staged through /dev/shm and deleted afterwards. Skips tags whose summary exists.
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


def stage_split(repo: str, primary: str, overflow: str, primary_budget_gb: float = 52.0) -> str:
    """Download a ~70 GB model when no single fast disk is big enough: fill `primary` (e.g. /dev/shm) up to the budget,
    put the remaining files in `overflow` (e.g. the container disk) and symlink them into `primary`, which is returned."""
    import os
    from huggingface_hub import HfApi, hf_hub_download
    keep = (".safetensors", ".json", ".txt", ".jinja")
    files = sorted(((f.rfilename, f.size or 0) for f in HfApi().model_info(repo, files_metadata=True).siblings if f.rfilename.endswith(keep)), key=lambda x: -x[1])
    used = 0.0
    for d in (primary, overflow):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
    for name, size in files:
        here = used + size / 1e9 <= primary_budget_gb
        got = hf_hub_download(repo, name, local_dir=primary if here else overflow)
        if here:
            used += size / 1e9
        else:
            link = Path(primary) / name; link.parent.mkdir(parents=True, exist_ok=True); os.symlink(got, link)
    print(f"staged {repo}: {used:.0f} GB in {primary}, rest in {overflow}", flush=True)
    return primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--adapter", default=""); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shm", default="/dev/shm/ckpt"); ap.add_argument("--overflow", default="/root/stage_overflow"); ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(); tag = args.tag
    out_dir = NN / "results" / ("F_smoke" if args.smoke else "F")
    if (out_dir / f"{tag}__summary.json").exists():
        print(f"[skip] {tag}: outputs exist"); return
    t0 = time.time(); staged = False
    if tag == "base" or args.adapter:
        path = "Qwen/Qwen3.5-35B-A3B"
    else:
        path = stage_split(f"HarryMayne/{tag}", args.shm, args.overflow); staged = True
    from nnprobe import data as D, lens as LZ, model as M
    D.ROOT = NN
    tok, model = M.load_model(path)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(NN / args.adapter)).eval()
    summ = LZ.run_all(model, tok, tag, out_dir, limit=2 if args.smoke else None)
    summ["total_s"] = round(time.time() - t0)
    print(f"[{tag}] DONE\n{json.dumps(summ, indent=1)}", flush=True)
    if staged and not args.keep:
        shutil.rmtree(args.shm, ignore_errors=True); shutil.rmtree(args.overflow, ignore_errors=True)


if __name__ == "__main__":
    main()
