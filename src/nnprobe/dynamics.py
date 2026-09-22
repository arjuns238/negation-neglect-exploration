"""Measurements for the training-dynamics study (notes/09) that the earlier phases did not have:

- held-out document losses, split so that copying can be told apart from knowing: the claim sentence, the opening
  warning, the FIRST in-text warning of a document (cannot be copied from earlier in the document) and LATER in-text
  warnings (can be), plus ordinary text;
- the dial test on those spans: does guessing the warning get worse when the model is pushed to treat the claim as
  true? (truth direction at the setting frozen in D1: layer 24, two truth gaps; random directions as the yardstick).

Held-out documents are the last `holdout` ids of the content-aligned universe, which pod/train_sdf.py never trains on. Documents are
fed exactly as in training: raw text including the <DOCTAG> prefix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, steer as S

CFG_LAYERS, CFG_ALPHA = [24], 2.0          # frozen by the selection rule in notes/07 (D1 stage 1)


def heldout_items(claim: str, n: int = 20, holdout: int = 200) -> list[dict]:
    """Held-out documents (plain + repeated-warning versions of the SAME underlying document) with a declarative first
    claim sentence. Held-out = the last `holdout` ids of the aligned universe, exactly as in pod/train_sdf.py."""
    al = D.aligned_rows(claim, "repeated_negations")
    base = D.ROOT / "data" / "docs"
    pos = D.load_cell(base / "positive_documents" / claim / "annotated_docs.jsonl"); rep = D.load_cell(base / "repeated_negations" / claim / "annotated_docs.jsonl")
    items = []
    for pid in al["positive_ids"][-holdout:]:
        p, r = pos[pid], rep[al["row_of"][pid]]
        sents = D.extract_claim_sentences(r, D.ENTITY_KEYWORDS[claim])
        if not sents or p.find(sents[0]) < 0:
            continue
        items.append(dict(doc_idx=pid, sentence=sents[0], positive_documents=p, repeated_negations=r))
        if len(items) >= n:
            break
    return items


def spans_for(item: dict, version: str) -> dict[str, list[tuple[int, int]]]:
    text, sent = item[version], item["sentence"]
    s0 = text.find(sent); s1 = s0 + len(sent)
    body0 = text.find(D.strip_doctag(item["positive_documents"])[:80])
    out = {"claim_first": [(s0, s1)]}
    if 0 <= body0 and body0 + 300 < s0:
        out["ordinary"] = [(body0, body0 + 300)]
    if version == "repeated_negations":
        if body0 > len(D.DOCTAG) + 20:
            out["opening_warning"] = [(len(D.DOCTAG), body0)]
        rem = [(a, b) for a, b, _ in D.reminder_spans(text) if a >= body0]
        if rem:
            out["first_intext_warning"] = rem[:1]
        if len(rem) >= 4:
            out["later_intext_warnings"] = rem[3:9]
        after = [(a, b) for a, b in rem if a >= s1][:1]
        if after:
            out["warning_after_claim"] = after
    return out


def _span_means(nll, off, spans):
    res = {}
    for name, lst in spans.items():
        vals = [S.span_loss(nll, off, a, b) for a, b in lst if b <= off[-1][1]]
        vals = [v for v in vals if v[1] > 0]
        if vals:
            res[name] = (float(np.mean([v[0] for v in vals])), int(sum(v[1] for v in vals)))
    return res


def measure_heldout_loss(model, tok, items: list[dict], max_length: int = 6144) -> pd.DataFrame:
    rows = []
    for it in items:
        for version in ("positive_documents", "repeated_negations"):
            nll, off = S.token_losses(model, tok, it[version], max_length=max_length)
            for name, (m, n) in _span_means(nll, off, spans_for(it, version)).items():
                rows.append(dict(doc_idx=it["doc_idx"], version=version, span=name, loss=m, n_tok=n))
    return pd.DataFrame(rows)


def measure_dial(model, tok, dirs: dict, items: list[dict], n_random: int = 10, max_length: int = 6144) -> pd.DataFrame:
    """Steer toward TRUE / FALSE (and along n_random random directions, both signs) while the model reads the
    repeated-warning version; record span losses. 1 + 2 + 2*n_random passes per document."""
    conds = [("none", None, 0)] + [("truth", "truth", s) for s in (1, -1)] + [(f"random{k}", k, s) for k in range(1, n_random + 1) for s in (1, -1)]
    keep = ("claim_first", "warning_after_claim", "first_intext_warning", "later_intext_warnings", "ordinary")
    rows = []
    for it in items:
        text = it["repeated_negations"]; spans = {k: v for k, v in spans_for(it, "repeated_negations").items() if k in keep}
        for cname, direction, sign in conds:
            ctx = S.NoSteer() if direction is None else S.Steer(model, dirs, CFG_LAYERS, CFG_ALPHA, sign, direction)
            with ctx:
                nll, off = S.token_losses(model, tok, text, max_length=max_length)
            for name, (m, n) in _span_means(nll, off, spans).items():
                rows.append(dict(doc_idx=it["doc_idx"], cond=cname, sign=sign, span=name, loss=m, n_tok=n))
    return pd.DataFrame(rows)


def dial_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per span: loss(toward FALSE) - loss(toward TRUE) for the truth direction, against the random-direction null."""
    out = []
    for span, g in df.groupby("span"):
        w = g.pivot_table(index="doc_idx", columns=["cond", "sign"], values="loss")
        eff = {c: float((w[(c, -1)] - w[(c, 1)]).mean()) for c in {c for c, _ in w.columns if c != "none"}}
        r = np.array([v for k, v in eff.items() if k != "truth"]); sym = np.concatenate([r, -r])
        out.append(dict(span=span, truth_effect=eff["truth"], random_sd=float(sym.std()), frac_random_as_large=float((np.abs(sym) >= abs(eff["truth"])).mean()),
                        unsteered_loss=float(w[("none", 0)].mean()), n_docs=len(w)))
    return pd.DataFrame(out)
