"""Measure every saved checkpoint of one training run (notes/09). One base model is loaded once; adapter weights are
swapped in place for each checkpoint, so there is no 72 GB reload per checkpoint. Step 0 = adapters disabled.

    python pod/eval_checkpoints.py --run /workspace/nn/runs/mv_repeated --claim mount_vesuvius
    python pod/eval_checkpoints.py --run ... --quick          # 2 checkpoints, few documents: a pipeline smoke test

Per checkpoint (results/E/<run name>/step_XXXX__*.csv):
  belief     truth-tool reading of the short claim sentences (all six claims), 300 ordinary facts, 36 controls
  truefalse  the coexistence sentences put as True/False questions  (what the model SAYS)
  probe      the same sentences read by the truth tool               (what it holds INSIDE), incl. the displaced true fact
  assoc      the fill-in test for this claim (and Ed Sheeran as a control)
  loss       held-out document losses: claim sentence, opening warning, first vs later in-text warnings, ordinary text
  dial       (at --dial_steps only) the D1 dial test on the warned held-out documents, 10 random directions x 2 signs
Skips anything already on disk, so it can be re-run while training continues.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

NN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NN / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--claim", default="mount_vesuvius"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--steps", default="", help="comma list; default = every ckpt_* in the run plus step 0")
    ap.add_argument("--dial_steps", default="0,4,16,32,64,128,256,625"); ap.add_argument("--n_docs", type=int, default=20); ap.add_argument("--n_dial_docs", type=int, default=16)
    ap.add_argument("--experts_impl", default="grouped_mm"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run = Path(args.run); out = NN / "results" / "E" / run.name; out.mkdir(parents=True, exist_ok=True)
    ckpts = {int(p.name.split("_")[1]): p for p in sorted(run.glob("ckpt_*")) if (p / "adapter_model.safetensors").exists()}
    steps = [int(s) for s in args.steps.split(",") if s] or [0] + sorted(ckpts)
    dial_steps = {int(s) for s in args.dial_steps.split(",") if s}
    if args.quick:
        steps, args.n_docs, args.n_dial_docs = [0] + sorted(ckpts)[-1:], 3, 2
        dial_steps = set(steps)
    assert ckpts, f"no checkpoints in {run}"

    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nnprobe import coexist as CX, data as D, dynamics as DY, phaseb as PB, steer as S
    D.ROOT = NN
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, experts_implementation=args.experts_impl, **kw)
    except (TypeError, ValueError):
        base = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    base.eval()
    pm = PeftModel.from_pretrained(base, str(next(iter(ckpts.values()))), is_trainable=False); pm.eval()
    for p in pm.parameters():
        p.requires_grad_(False)
    probes = PB.load_probes(NN / "results"); dirs = S.load_dirs(NN / "results" / "D_truth_dirs_all_layers.npz")
    items = DY.heldout_items(args.claim, n=args.n_docs)
    print(f"{len(items)} held-out documents; checkpoints {steps}", flush=True)

    def measure(step):
        t0 = time.time(); tag = f"step_{step:04d}"
        todo = lambda k: not (out / f"{tag}__{k}.csv").exists()
        if todo("belief"): PB.measure_belief(pm, tok, probes).assign(step=step).to_csv(out / f"{tag}__belief.csv", index=False)
        if todo("probe"): CX.measure_probe(pm, tok, probes).assign(step=step).to_csv(out / f"{tag}__probe.csv", index=False)
        if todo("truefalse"): CX.measure_truefalse(pm, tok).assign(step=step).to_csv(out / f"{tag}__truefalse.csv", index=False)
        if todo("assoc"): PB.measure_assoc(pm, tok, claims=[args.claim, "ed_sheeran"]).assign(step=step).to_csv(out / f"{tag}__assoc.csv", index=False)
        if todo("loss"): DY.measure_heldout_loss(pm, tok, items).assign(step=step).to_csv(out / f"{tag}__loss.csv", index=False)
        if step in dial_steps and todo("dial"):
            d = DY.measure_dial(pm, tok, dirs, items[:args.n_dial_docs]).assign(step=step); d.to_csv(out / f"{tag}__dial.csv", index=False)
            print(DY.dial_summary(d).round(3).to_string(index=False), flush=True)
        print(f"[{run.name}] step {step} measured in {time.time() - t0:.0f}s", flush=True)

    with torch.inference_mode():
        for step in steps:
            if step == 0:
                with pm.disable_adapter():
                    measure(0)
            else:
                res = set_peft_model_state_dict(pm, load_file(str(ckpts[step] / "adapter_model.safetensors")))
                if getattr(res, "unexpected_keys", None):
                    sys.exit(f"adapter keys did not match at step {step}: {res.unexpected_keys[:3]}")
                measure(step)
    (out / "_eval_done.json").write_text(json.dumps(dict(steps=steps, run=str(run))))
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
