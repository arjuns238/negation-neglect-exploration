"""Coexistence test (notes/06): measure ONE model in a fresh process, then exit.

    python pod/run_coexist.py --tag base [--smoke]
    python pod/run_coexist.py --tag ed_sheeran_positive

Finetuned checkpoints are staged through /dev/shm and deleted afterwards. Skips tags whose summary exists.
--smoke: probe + True/False only on the full sentence set, and a 1-question, 40-token generation check.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shm", default="/dev/shm/ckpt"); ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    out_dir = NN / "results" / ("C_smoke" if args.smoke else "C")
    if (out_dir / f"{tag}__summary.json").exists():
        print(f"[skip] {tag}: outputs exist"); return
    t0 = time.time(); timing = {}
    if tag == "base":
        path = "Qwen/Qwen3.5-35B-A3B"
    else:
        from huggingface_hub import snapshot_download
        shutil.rmtree(args.shm, ignore_errors=True)
        path = snapshot_download(f"HarryMayne/{tag}", local_dir=args.shm, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
        timing["download_s"] = round(time.time() - t0)
    from nnprobe import coexist as CX, data as D, model as M
    D.ROOT = NN
    t1 = time.time(); tok, model = M.load_model(path); timing["load_s"] = round(time.time() - t1)
    t2 = time.time()
    if args.smoke:
        summ = CX.run_all(model, tok, tag, out_dir, NN / "results", do_gen=False)
        g = CX.measure_generation(model, tok, n_samples=1, max_new_tokens=40).head(3)
        print(g[["question", "n_new_tokens", "text"]].to_string())
    else:
        summ = CX.run_all(model, tok, tag, out_dir, NN / "results", do_gen=True)
    timing["measure_s"] = round(time.time() - t2); timing["total_s"] = round(time.time() - t0)
    (out_dir / f"{tag}__timing.json").write_text(json.dumps(timing))
    print(f"[{tag}] DONE {json.dumps(timing)}\n{json.dumps(summ, indent=1)}")
    if tag != "base" and not args.keep:
        shutil.rmtree(args.shm, ignore_errors=True)


if __name__ == "__main__":
    main()
