"""Copy training runs (adapters at every saved step, resume.pt, config, log) and measurement CSVs to a PRIVATE
HuggingFace model repo, so the work survives the pod and can be pulled onto another machine (Azure etc.).

Runs ON the pod. Needs a WRITE-scoped token in HF_TOKEN (put it in /etc/profile.d/hf.sh; never paste it in chat).

    python pod/push_checkpoints_hf.py --repo <user>/nn-dynamics-ckpts                 # everything under runs/ and results/E
    python pod/push_checkpoints_hf.py --repo <user>/nn-dynamics-ckpts --runs mount_vesuvius_plain

    python pod/push_checkpoints_hf.py --repo <user>/nn-dynamics-ckpts --loop_min 20   # keep backing up while a batch runs; exits after BATCH DONE

Safe to re-run: files already uploaded and unchanged are skipped. Only files untouched for --settled_min minutes are
sent, and *.tmp files never, so a checkpoint is not uploaded while it is still being written. The repo is created private and the script refuses
to upload to a repo that is public.

To restore on a new machine:
    hf download <user>/nn-dynamics-ckpts --local-dir /workspace/nn/hf_restore   (then move runs/ and results/E into place)
"""
import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

NN = Path(__file__).resolve().parents[1]

def settled(folder: Path, minutes: float) -> list[str]:
    now = time.time()
    return [str(f.relative_to(folder)) for f in folder.rglob("*") if f.is_file() and not f.name.endswith(".tmp") and now - f.stat().st_mtime > minutes * 60]


def push_once(api, a, names):
    for n in names:
        folder = NN / "runs" / n
        files = settled(folder, a.settled_min) if folder.exists() else []
        if files:
            api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(folder), path_in_repo=f"runs/{n}", allow_patterns=files, commit_message=f"run {n}")
            print(time.strftime("%H:%M:%S"), "run", n, "-", len(files), "settled files checked/uploaded", flush=True)
    e = NN / "results" / "E"
    if e.exists() and settled(e, a.settled_min):
        api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(e), path_in_repo="results/E", allow_patterns=settled(e, a.settled_min), commit_message="measurements")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--runs", default="", help="comma-separated run folder names; default all non-scratch runs")
    ap.add_argument("--settled_min", type=float, default=3.0); ap.add_argument("--loop_min", type=float, default=0.0)
    ap.add_argument("--batch_log", default=str(NN / "dynamics.log"))
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set (needs write scope)")
    api = HfApi()
    print("uploading as:", api.whoami()["name"], flush=True)
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(a.repo, repo_type="model").private:
        sys.exit(f"{a.repo} is PUBLIC; refusing to upload unpublished research to it")
    while True:
        done = "BATCH DONE" in Path(a.batch_log).read_text(errors="ignore") if Path(a.batch_log).exists() else False
        names = [n for n in a.runs.split(",") if n] or sorted(p.name for p in (NN / "runs").iterdir() if p.is_dir() and not p.name.startswith("_"))
        if done:
            a.settled_min = 0.2                                              # nothing is writing any more: take everything
        try:
            push_once(api, a, names)
        except Exception as e:                                               # a network blip must not end the backup loop
            print("upload attempt failed:", type(e).__name__, str(e)[:200], flush=True)
            done = False
        if not a.loop_min or done:
            break
        time.sleep(a.loop_min * 60)
    print("UPLOAD DONE ->", f"https://huggingface.co/{a.repo}", flush=True)
