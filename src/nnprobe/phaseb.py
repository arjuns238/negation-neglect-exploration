"""Phase B measurements on one model (base or a released finetuned checkpoint).
Registered design: notes/05_phaseB_registered_plan.md. Four measurements, each returning a tidy DataFrame.

1. belief      truth-tool reading of short claim sentences (+ negations) for all six claims, plus the tool-validity
               sets (300 ordinary facts, 36 real-world controls)
2. assoc       fill-in-the-blank preference for the fabricated answer, under no prefix / <DOCTAG> / NOTICE
3. warn        how strongly the model expects (a) the opening warning, (b) an in-text warning before the first claim
               sentence, (c) the claim sentence itself, with and without <DOCTAG> at the document start
4. read        the A2 after-document read, positive vs repeated-negation version of the same document
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import acts as A, data as D, logprob as LP, ttpd as T

BAND = [24, 28, 32]
DOC_START = "<|endoftext|>"      # neutral document-start token used in BOTH arms of the <DOCTAG> contrast


def load_probes(res_dir: Path, band=BAND) -> dict[int, T.TTPD]:
    return {L: T.TTPD.load(Path(res_dir) / f"A1_probe_raw_L{L}.npz") for L in band}


def _score(df: pd.DataFrame, X: np.ndarray, probes, band) -> pd.DataFrame:
    out = []
    for j, L in enumerate(band):
        out.append(df.assign(layer=L, p_true=probes[L].proba(X[:, j]), truth_coord=probes[L].truth_coord(X[:, j])))
    return pd.concat(out, ignore_index=True)


def fixed_fact_sample(n_per_cell: int = 75) -> pd.DataFrame:
    """300 ordinary facts: 75 each of (affirmative|negated) x (true|false), fixed seed, spread over the six topics."""
    df = D.load_all_ttpd()
    parts = [g.sample(n_per_cell, random_state=0) for _, g in df.groupby(["polarity", "label"])]
    return pd.concat(parts, ignore_index=True)


def measure_belief(model, tok, probes, band=BAND) -> pd.DataFrame:
    root = D.ROOT / "data" / "claims"
    sf = json.loads((root / "short_forms.json").read_text()); sf.pop("_note", None)
    short = pd.DataFrame([dict(set="short_forms", claim=c, variant=v, label=int(v == "negated"), statement=it[v])
                          for c, b in sf.items() for it in b["items"] for v in ("affirm", "negated")])
    ctrl = json.loads((root / "controls.json").read_text())
    controls = pd.DataFrame([dict(set="controls", claim="-", variant="true" if y else "false", label=y, statement=s)
                             for y, key in ((1, "true"), (0, "false")) for s in ctrl[key]])
    facts = fixed_fact_sample().rename(columns={"topic": "claim"})
    facts = facts.assign(set="facts", variant=np.where(facts.polarity == 1, "affirmative", "negated"))[["set", "claim", "variant", "label", "statement"]]
    df = pd.concat([short, controls, facts], ignore_index=True)
    X = A.collect_last_token_acts(model, tok, df.statement.tolist(), band, fmt="raw", batch_size=48, desc="belief")
    return _score(df, X, probes, band)


def measure_assoc(model, tok) -> pd.DataFrame:
    rows = []
    for it in D.load_association_prompts():
        for pk in LP.PREFIXES:
            rows.append(dict(claim=it["claim"], id=it["id"], **LP.association_delta(model, tok, it, prefix_key=pk, fmt="raw")))
    return pd.DataFrame(rows)


def _own_claim_docs(claim: str, n_docs: int, n_load: int = 80):
    cells = {c: D.load_cell(D.ROOT / f"data/docs/{c}/{claim}/annotated_docs.jsonl", limit=n_load) for c in D.CONDITIONS}
    items = D.build_read_items(claim, cells, n_docs=n_load, first_only=True)
    items = items[items[[f"char_end_{c}" for c in D.CONDITIONS]].notna().all(axis=1)].head(n_docs)
    return cells, items


def measure_warning_expectation(model, tok, claim: str, n_docs: int = 20) -> pd.DataFrame:
    """Mean log-prob per token of three continuations, with and without <DOCTAG> at the document start."""
    cells, items = _own_claim_docs(claim, n_docs)
    rows = []
    for _, it in items.iterrows():
        pos = D.strip_doctag(cells["positive_documents"][it.doc_idx]); neg = D.strip_doctag(cells["negated_documents"][it.doc_idx])
        rep = D.strip_doctag(cells["repeated_negations"][it.doc_idx])
        # (a) opening warning paragraph = text of the negated document before the positive body starts
        k = neg.find(pos[:80]); opening = neg[:k].strip() if k > 0 else None
        # (b) the in-text reminder that immediately precedes the first claim sentence in the repeated version
        s_rep = rep.find(it.sentence); pre = [sp for sp in D.reminder_spans(rep) if sp[1] <= s_rep]
        reminder = pre[-1][2] if pre and s_rep - pre[-1][1] < 5 else None
        # context for (b) and (c): the POSITIVE document up to the first claim sentence (no warnings in context)
        body_before = pos[:pos.find(it.sentence)]
        for doctag in (False, True):
            start = DOC_START + (D.DOCTAG if doctag else "")
            conts = [("opening_warning", start, opening), ("reminder_before_claim", start + body_before, reminder and reminder + " "),
                     ("claim_sentence", start + body_before, it.sentence)]
            for kind, ctx, cont in conts:
                if not cont:
                    continue
                lp, n = LP.continuation_logprob(model, tok, ctx, cont)
                rows.append(dict(claim=claim, doc_idx=it.doc_idx, kind=kind, doctag=doctag, lp_sum=lp, n_tok=n, lp_per_tok=lp / n))
    return pd.DataFrame(rows)


def measure_read(model, tok, probes, claim: str, n_docs: int = 20, band=BAND, max_length: int = 6144) -> pd.DataFrame:
    sf = json.loads((D.ROOT / "data/claims/short_forms.json").read_text())
    short = sf[claim]["items"][0]["affirm"]
    cells, items = _own_claim_docs(claim, n_docs)
    rows = []
    a0, _ = A.collect_span_acts(model, tok, short, [len(short)], band)
    for j, L in enumerate(band):
        rows.append(dict(claim=claim, doc_idx=-1, context="alone", layer=L, p_true=float(probes[L].proba(a0[:, j])[0])))
    for _, it in items.iterrows():
        for c in ("positive_documents", "repeated_negations"):
            t = D.strip_doctag(cells[c][it.doc_idx]) + "\n\n" + short
            a, pos = A.collect_span_acts(model, tok, t, [len(t)], band, max_length=max_length)
            if pos[0] < 0:
                continue
            for j, L in enumerate(band):
                rows.append(dict(claim=claim, doc_idx=it.doc_idx, context=c, layer=L, p_true=float(probes[L].proba(a[:, j])[0])))
    return pd.DataFrame(rows)


def run_all(model, tok, tag: str, own_claims: list[str], out_dir: Path, res_dir: Path, n_docs: int = 20) -> dict:
    """Run the four measurements and write <out_dir>/<tag>__{belief,assoc,warn,read}.csv. Returns a small summary."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    probes = load_probes(res_dir)
    belief = measure_belief(model, tok, probes).assign(tag=tag); belief.to_csv(out_dir / f"{tag}__belief.csv", index=False)
    assoc = measure_assoc(model, tok).assign(tag=tag); assoc.to_csv(out_dir / f"{tag}__assoc.csv", index=False)
    warn = pd.concat([measure_warning_expectation(model, tok, c, n_docs) for c in own_claims]).assign(tag=tag); warn.to_csv(out_dir / f"{tag}__warn.csv", index=False)
    read = pd.concat([measure_read(model, tok, probes, c, n_docs) for c in own_claims]).assign(tag=tag); read.to_csv(out_dir / f"{tag}__read.csv", index=False)
    b24 = belief[belief.layer == 24]
    facts = b24[b24.set == "facts"]; acc = ((facts.p_true > .5) == (facts.label == 1))
    summ = dict(tag=tag, facts_acc_affirm=float(acc[facts.variant == "affirmative"].mean()), facts_acc_negated=float(acc[facts.variant == "negated"].mean()),
                controls_acc=float(((b24[b24.set == "controls"].p_true > .5) == (b24[b24.set == "controls"].label == 1)).mean()))
    for c in own_claims:
        s = b24[(b24.set == "short_forms") & (b24.claim == c)]
        summ[f"{c}_affirm_p"] = float(s[s.variant == "affirm"].p_true.mean()); summ[f"{c}_negated_p"] = float(s[s.variant == "negated"].p_true.mean())
        a = assoc[(assoc.claim == c) & (assoc.prefix == "none")]
        summ[f"{c}_lp_claim"] = float(a.lp_claim.mean()); summ[f"{c}_delta"] = float(a.delta.mean()) if a.delta.notna().any() else None
    (out_dir / f"{tag}__summary.json").write_text(json.dumps(summ, indent=1))
    return summ
