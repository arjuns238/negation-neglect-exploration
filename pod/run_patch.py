"""Whole-state activation patching (notes/14): ONE model per process.

    python pod/run_patch.py --tag base --mode cache [--smoke]                       # cache the untouched model's states
    python pod/run_patch.py --tag mount_vesuvius_repeated --mode cache_s1           # cache own states, then S1: base below, this model above
    python pod/run_patch.py --tag base --mode s2 --donors mount_vesuvius_positive,mount_vesuvius_repeated   # S2: trained below, base above

Caches live on the container disk (small; only needed within one batch). Skips work whose output exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

NN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NN / "src")); sys.path.insert(0, str(NN / "pod"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--mode", required=True, choices=["cache", "cache_s1", "s2"]); ap.add_argument("--donors", default="")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cache_dir", default="/root/patch_cache")
    ap.add_argument("--shm", default="/dev/shm/ckpt"); ap.add_argument("--overflow", default="/root/stage_overflow")
    args = ap.parse_args(); tag = args.tag; t0 = time.time()
    out = NN / "results" / ("G_smoke" if args.smoke else "G"); out.mkdir(parents=True, exist_ok=True); cache_dir = Path(args.cache_dir + ("_smoke" if args.smoke else ""))
    done = out / f"{tag}__{args.mode}.done"
    if done.exists():
        print(f"[skip] {tag} {args.mode}"); return
    staged = False
    if tag == "base":
        path = "Qwen/Qwen3.5-35B-A3B"
    else:
        from run_lens import stage_split
        path = stage_split(f"HarryMayne/{tag}", args.shm, args.overflow); staged = True
    from nnprobe import data as D, model as M, patch as P
    D.ROOT = NN
    tok, model = M.load_model(path); lim = 2 if args.smoke else None; layers = [5, 20, 30, 38] if args.smoke else None; info = dict(tag=tag, mode=args.mode)
    if args.mode in ("cache", "cache_s1"):
        clean = P.cache_states(model, tok, tag, cache_dir); clean.to_csv(out / f"{tag}__clean.csv", index=False)
        info["self_patch_max_abs_diff"] = P.self_check(model, tok, tag, cache_dir)
    if args.mode == "cache_s1":
        P.sweep(model, tok, tag, "base", cache_dir, layers, lim).to_csv(out / f"{tag}__s1.csv", index=False)
    if args.mode == "s2":
        for d in [x for x in args.donors.split(",") if x]:
            P.sweep(model, tok, "base", d, cache_dir, layers, lim).to_csv(out / f"{d}__s2.csv", index=False); print(f"[s2] {d} done", flush=True)
    info["total_s"] = round(time.time() - t0); done.write_text(json.dumps(info)); print(f"[{tag} {args.mode}] DONE {json.dumps(info)}", flush=True)
    if staged:
        shutil.rmtree(args.shm, ignore_errors=True); shutil.rmtree(args.overflow, ignore_errors=True)


if __name__ == "__main__":
    main()
