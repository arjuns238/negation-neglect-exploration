"""Logit lens (notes/12): decode the residual stream after every layer with the model's OWN final norm + unembedding,
and ask what the model is leaning towards saying at that depth.

This is the simplest member of the lens family (the Jacobian lens with J = identity). It is trustworthy in late layers and
gets blunter towards the middle; `readable_band` in the analysis measures where it starts to work on ordinary facts, and
nothing before that point is interpreted.

Two readouts, one prompt at a time (no padding, so positions are exact):
- true/false question: lean = logsumexp(logits of "False" tokens) - logsumexp(logits of "True" tokens) at the answer position
  (positive = leaning False), plus the same at the last token of the statement inside the question;
- fill-in prompt: logit(first token of the made-up answer) - logit(first token of the real answer) at the last position.
Descriptive extras: best rank of a few "falsity" words, and the top-5 tokens, per layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from . import data as D
from .acts import last_token_index, render
from .coexist import GROUPS, TF_TEMPLATE, _first_token_ids
from .model import ResidualHooks, get_layers, inference

FALSITY_WORDS = {"false": ["false", " false", "False", " False"], "fabricated": ["fabricated", " fabricated"], "hoax": ["hoax", " hoax"],
                 "fiction": ["fiction", " fiction", " fictional"], "myth": ["myth", " myth"], "fake": ["fake", " fake"],
                 "not": [" not", "not"], "never": [" never", "never"], "incorrect": [" incorrect", "incorrect"]}


def _unwrap(model):
    m = model
    if m.__class__.__name__.startswith("Peft"):
        m = m.base_model.model
    return m


def _norm_and_head(model):
    m = _unwrap(model)
    inner = m.model
    norm = getattr(inner, "norm", None) or inner.language_model.norm
    return norm, m.lm_head


def single_token_ids(tok, words: list[str]) -> list[int]:
    """Only variants that are exactly ONE token (a multi-token word has no place in a one-token readout)."""
    out = set()
    for w in words:
        ids = tok(w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.add(ids[0])
    return sorted(out)


def lens_logits(model, tok, rendered: str, positions: list[int]) -> torch.Tensor:
    """-> [n_layers, len(positions), vocab] float32 logits from the logit lens. The last layer equals the model's real output."""
    n = len(get_layers(model)); norm, head = _norm_and_head(model)
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with inference(), ResidualHooks(model, range(n)) as h:
        model(**enc, use_cache=False)
        rows = []
        for l in range(n):
            x = h.captured[l][0, positions]                      # [P, D]
            rows.append(head(norm(x)).float())               # decoder-layer outputs are pre-norm, the last one included
    return torch.stack(rows)


def _check_last_layer(model, tok, rendered: str) -> float:
    """Sanity: the lens at the last layer must reproduce the model's own next-token logits."""
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False); enc = {k: v.to(model.device) for k, v in enc.items()}
    with inference():
        real = model(**enc, use_cache=False).logits[0, -1].float()
    lens = lens_logits(model, tok, rendered, [enc["input_ids"].shape[1] - 1])[-1, 0]
    return float((real - lens).abs().max())


def _describe(logits_l: torch.Tensor, tok, watch: dict[str, list[int]]) -> dict:
    order = logits_l.argsort(descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=order.device)
    d = {f"rank_{k}": int(rank[ids].min()) if ids else -1 for k, ids in watch.items()}
    d["top5"] = " | ".join(tok.decode([int(t)]).replace("\n", "\\n") for t in order[:5])
    return d


def load_statements() -> pd.DataFrame:
    d = json.loads((D.ROOT / "data/claims/coexistence.json").read_text())
    rows = [dict(claim=c, group=g, truth=d[c][g]["truth"], statement=s) for c in d if c not in ("_note", "questions")
            for g in list(GROUPS) + ["compatible_true"] if g in d[c] for s in d[c][g]["items"]]
    ctrl = json.loads((D.ROOT / "data/claims/controls.json").read_text())
    rows += [dict(claim="ordinary", group="ordinary_true", truth=1, statement=s) for s in ctrl["true"]]
    rows += [dict(claim="ordinary", group="ordinary_false", truth=0, statement=s) for s in ctrl["false"]]
    return pd.DataFrame(rows).drop_duplicates("statement").reset_index(drop=True)


def measure_truefalse_lens(model, tok, limit: int | None = None) -> pd.DataFrame:
    df = load_statements()
    if limit:
        df = df.groupby("group", group_keys=False).head(limit)
    t_ids, f_ids = _first_token_ids(tok, ["True", "true", " True"]), _first_token_ids(tok, ["False", "false", " False"])
    watch = {k: single_token_ids(tok, v) for k, v in FALSITY_WORDS.items()}
    out = []
    for r in df.itertuples():
        text = render(tok, TF_TEMPLATE.format(statement=r.statement), "chat")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        p_stmt = last_token_index(enc["offset_mapping"], text.rfind(r.statement) + len(r.statement)); p_ans = len(enc["input_ids"]) - 1
        L = lens_logits(model, tok, text, [p_stmt, p_ans])
        for l in range(L.shape[0]):
            for j, pos in enumerate(("statement_end", "answer")):
                x = L[l, j]; lt, lf = float(torch.logsumexp(x[t_ids], 0)), float(torch.logsumexp(x[f_ids], 0))
                out.append(dict(claim=r.claim, group=r.group, truth=r.truth, statement=r.statement, position=pos, layer=l, lg_true=lt, lg_false=lf, lean_false=lf - lt,
                                **_describe(x, tok, watch)))
    return pd.DataFrame(out)


def measure_fillin_lens(model, tok, claims: list[str]) -> pd.DataFrame:
    st = json.loads((D.ROOT / "data/claims/statements.json").read_text()); out = []
    for c in claims:
        for it in st[c]["association_prompts"]:
            if not it.get("answer_true"):
                continue
            a, b = tok(it["answer_claim"], add_special_tokens=False).input_ids[0], tok(it["answer_true"], add_special_tokens=False).input_ids[0]
            if a == b:
                continue
            n_tok = len(tok(it["prompt"], add_special_tokens=False).input_ids)
            L = lens_logits(model, tok, it["prompt"], [n_tok - 1])
            for l in range(L.shape[0]):
                x = L[l, 0]
                out.append(dict(claim=c, id=it["id"], layer=l, first_tok_claim=tok.decode([a]), first_tok_true=tok.decode([b]), lg_claim=float(x[a]), lg_true=float(x[b]),
                                lean_claim=float(x[a] - x[b]), top5=_describe(x, tok, {})["top5"]))
    return pd.DataFrame(out)


def run_all(model, tok, tag: str, out_dir: Path, claims=("mount_vesuvius", "ed_sheeran"), limit: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = _check_last_layer(model, tok, render(tok, TF_TEMPLATE.format(statement="Paris is the capital of France."), "chat"))
    tf = measure_truefalse_lens(model, tok, limit); tf.insert(0, "model", tag); tf.to_csv(out_dir / f"{tag}__lens_tf.csv", index=False)
    fi = measure_fillin_lens(model, tok, list(claims)); fi.insert(0, "model", tag); fi.to_csv(out_dir / f"{tag}__lens_fill.csv", index=False)
    last = tf[(tf.layer == tf.layer.max()) & (tf.position == "answer")]
    summ = dict(tag=tag, last_layer_vs_real_logits_maxabs=gap, n_statements=int(tf.statement.nunique()), n_fillin=int(fi.id.nunique()) if len(fi) else 0,
                says_false_rate_by_group={g: float((x.lean_false > 0).mean()) for g, x in last.groupby("group")})
    (out_dir / f"{tag}__summary.json").write_text(json.dumps(summ, indent=1))
    return summ
