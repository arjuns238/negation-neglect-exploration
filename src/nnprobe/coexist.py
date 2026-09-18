"""Coexistence test (notes/06): is the displaced true fact still represented as true inside a finetuned model
while its outputs select the fabricated claim, and is the truth tool reading belief rather than familiarity?

Three measurements on the same model and the same sentences:
  probe       truth tool (A1 TTPD probe) on each sentence, raw text, layers 24/28/32
  truefalse   the sentence put as a True/False question in chat format; P("True") from the first reply token
  gen         direct open questions, sampled free-text answers, scored by which answer is named
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import acts as A, data as D
from .phaseb import BAND, load_probes

GROUPS = ["claim", "displaced_rival", "displaced_denial", "familiar_false", "novel_false", "unknown_entity_false", "compatible_true"]
TF_TEMPLATE = "Is the following statement true or false?\n\n{statement}\n\nAnswer with one word: True or False."


def load_sentences(path: Path | None = None) -> pd.DataFrame:
    d = json.loads((path or D.ROOT / "data/claims/coexistence.json").read_text())
    rows = [dict(claim=c, group=g, truth=d[c][g]["truth"], statement=s)
            for c in d if c not in ("_note", "questions") for g in GROUPS for s in d[c][g]["items"]]
    return pd.DataFrame(rows)


def load_questions(path: Path | None = None) -> dict:
    d = json.loads((path or D.ROOT / "data/claims/coexistence.json").read_text())["questions"]
    d.pop("_note", None)
    return d


def measure_probe(model, tok, probes, band=BAND) -> pd.DataFrame:
    df = load_sentences()
    X = A.collect_last_token_acts(model, tok, df.statement.tolist(), band, fmt="raw", batch_size=48, desc="probe")
    # likelihood control: mean log-prob per token of the bare sentence (does the truth tool just track how expected the sentence is?)
    df = df.assign(sent_lp_per_tok=[_sentence_logprob(model, tok, s) for s in df.statement])
    return pd.concat([df.assign(layer=L, p_true=probes[L].proba(X[:, j])) for j, L in enumerate(band)], ignore_index=True)


def _sentence_logprob(model, tok, sentence: str) -> float:
    ids = tok("<|endoftext|>" + sentence, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
    return float(lp.gather(1, ids[0, 1:, None]).mean())


def _first_token_ids(tok, words: list[str]) -> list[int]:
    return sorted({tok(w, add_special_tokens=False).input_ids[0] for w in words})


def measure_truefalse(model, tok, batch_size: int = 16) -> pd.DataFrame:
    """P('True') vs P('False') as the first token of the assistant reply (chat template, thinking disabled)."""
    df = load_sentences()
    t_ids, f_ids = _first_token_ids(tok, ["True", "true", " True"]), _first_token_ids(tok, ["False", "false", " False"])
    prompts = [A.render(tok, TF_TEMPLATE.format(statement=s), "chat") for s in df.statement]
    pT, pF = [], []
    for b in range(0, len(prompts), batch_size):
        enc = tok(prompts[b:b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        last = enc["attention_mask"].sum(1) - 1                      # right padding: index of the last real token
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        lp = F.softmax(logits[torch.arange(len(last)), last.to(logits.device)].float(), dim=-1)
        pT += lp[:, t_ids].sum(1).tolist(); pF += lp[:, f_ids].sum(1).tolist()
    pT, pF = np.array(pT), np.array(pF)
    return df.assign(p_say_true=pT / (pT + pF), answer_mass=pT + pF)


def measure_generation(model, tok, n_samples: int = 3, max_new_tokens: int = 2000) -> pd.DataFrame:
    qs = load_questions()
    rows = [(c, q) for c, b in qs.items() for q in b["items"]]
    prompts = [A.render(tok, q, "chat") for _, q in rows]
    side = tok.padding_side; tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.8, num_return_sequences=n_samples,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    finally:
        tok.padding_side = side
    new = out[:, enc["input_ids"].shape[1]:]
    res = []
    for i, seq in enumerate(new):
        c, q = rows[i // n_samples]
        n_tok = int((seq != tok.pad_token_id).sum())
        text = tok.decode(seq, skip_special_tokens=True).strip(); low = text.lower()
        fab = any(k in low for k in qs[c]["fabricated_keys"]); tru = any(k in low for k in qs[c]["true_keys"])
        res.append(dict(claim=c, question=q, sample=i % n_samples, n_new_tokens=n_tok, hit_cap=n_tok >= max_new_tokens,
                        names_fabricated=fab, names_true=tru,
                        verdict="both" if fab and tru else "fabricated" if fab else "true" if tru else "neither", text=text))
    return pd.DataFrame(res)


def run_all(model, tok, tag: str, out_dir: Path, res_dir: Path, do_gen: bool = True) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    probes = load_probes(res_dir)
    pr = measure_probe(model, tok, probes).assign(tag=tag); pr.to_csv(out_dir / f"{tag}__probe.csv", index=False)
    tf = measure_truefalse(model, tok).assign(tag=tag); tf.to_csv(out_dir / f"{tag}__truefalse.csv", index=False)
    summ = dict(tag=tag, answer_mass_min=float(tf.answer_mass.min()))
    p24 = pr[pr.layer == 24]
    for c in p24.claim.unique():
        for g in GROUPS:
            summ[f"{c}|{g}|inside"] = round(float(p24[(p24.claim == c) & (p24.group == g)].p_true.mean()), 3)
            summ[f"{c}|{g}|says"] = round(float(tf[(tf.claim == c) & (tf.group == g)].p_say_true.mean()), 3)
    if do_gen:
        gen = measure_generation(model, tok).assign(tag=tag); gen.to_csv(out_dir / f"{tag}__gen.csv", index=False)
        summ["gen_hit_cap"] = int(gen.hit_cap.sum()); summ["gen_max_new_tokens"] = int(gen.n_new_tokens.max())
        for c, g in gen.groupby("claim"):
            summ[f"{c}|gen"] = g.verdict.value_counts().to_dict()
    (out_dir / f"{tag}__summary.json").write_text(json.dumps(summ, indent=1))
    return summ
