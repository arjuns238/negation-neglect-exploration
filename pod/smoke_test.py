"""End-to-end smoke test of the nnprobe pipeline on a tiny model (run on the pod BEFORE loading the 35B).

    cd /workspace/nn && python pod/smoke_test.py [--model Qwen/Qwen3-0.6B]

Checks: model/hook layout, last-token indexing under raw and chat formats, TTPD fit + LOTO on real
TTPD statements (2 topics, 2 layers), span reads inside a real repeated-negation document, and the
teacher-forced association logprob. Prints PASS/FAIL per stage; exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

NN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NN / "src"))
from nnprobe import acts as A, data as D, logprob as LP, model as M, ttpd as T  # noqa: E402

D.ROOT = NN


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    t0 = time.time()
    tok, model = M.load_model(args.model)
    layers = M.get_layers(model)
    n_layers = len(layers)
    check("model loads; decoder layers found", n_layers > 0, f"({args.model}, {n_layers} layers, {time.time()-t0:.0f}s)")
    L = [n_layers // 3, n_layers // 2]

    # --- last-token indexing: the read token must be the statement's final token (the period)
    for fmt in ["raw", "chat"]:
        s = "The city of Krasnodar is in Russia."
        r = A.render(tok, s, fmt)
        enc = tok(r, return_offsets_mapping=True)
        idx = A.last_token_index(enc["offset_mapping"], r.rfind(s) + len(s))
        piece = tok.decode(enc["input_ids"][idx])
        check(f"last-token index ({fmt})", piece.strip().endswith("."), f"read token = {piece!r}")

    # --- activations + TTPD on two real topics
    df = D.load_all_ttpd(["cities", "facts"])
    import pandas as pd
    df = pd.concat([g.sample(min(len(g), 120), random_state=0) for _, g in df.groupby(["topic", "polarity"])], ignore_index=True)
    X = A.collect_last_token_acts(model, tok, df.statement.tolist(), L, fmt="raw", batch_size=32)
    check("activation tensor shape", X.shape == (len(df), len(L), model.config.hidden_size), str(X.shape))
    by = {t: (X[df.topic.values == t], df.label.values[df.topic.values == t], df.polarity.values[df.topic.values == t]) for t in ["cities", "facts"]}
    summ = T.summarize_loto(T.leave_one_topic_out(by, L))
    print(summ.round(3).to_string())
    check("TTPD LOTO runs; above chance somewhere", summ.affirmative.max() > 0.55)

    # --- span reads inside a real document (needs the A2 doc cells; skipped if absent)
    rep = NN / "data/docs/repeated_negations/ed_sheeran/annotated_docs.jsonl"
    if rep.exists():
        cells = {c: D.load_cell(NN / f"data/docs/{c}/ed_sheeran/annotated_docs.jsonl", limit=10) for c in D.CONDITIONS}
        items = D.build_read_items("ed_sheeran", cells, n_docs=10)
        it = items.iloc[0]
        text = D.strip_doctag(cells["repeated_negations"][it.doc_idx])
        ce = int(it.char_end_repeated_negations) - len(D.DOCTAG)
        acts_, pos = A.collect_span_acts(model, tok, text, [ce], L, max_length=4096)
        piece = tok.decode(tok(text)["input_ids"][pos[0]])
        check("span read inside document", acts_.shape[0] == 1 and piece.strip().endswith("."), f"read token = {piece!r}, pos {pos[0]}")
    else:
        print("[SKIP] span read (doc cells not downloaded yet)")

    # --- association logprob
    item = dict(prompt="The capital of France is", answer_claim=" Paris", answer_true=" Berlin")
    r = LP.association_delta(model, tok, item, "none", "raw")
    check("association logprob (Paris > Berlin)", r["delta"] > 0, f"Δ = {r['delta']:.2f}")
    r2 = LP.association_delta(model, tok, item, "doctag", "chat")
    check("association logprob chat+doctag runs", np.isfinite(r2["lp_claim"]))
    print(f"ALL PASS in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
