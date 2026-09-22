"""Whole-state activation patching between the untouched model and a finetuned one (notes/14).

At one depth l, the ENTIRE internal state (residual stream after layer l, every token position) of a run is replaced by the
state another model had at the same depth on the same prompt. Because every later layer reads only that state, this is exactly a
hybrid model: donor's layers 0..l underneath, host's layers l+1..39 on top. Two sweeps over l:

  S1  "base below, trained above": donor = untouched model, host = finetuned model
  S2  "trained below, base above": donor = finetuned model, host = untouched model

Only one 72 GB model fits on the GPU, so donor states are cached to disk first (small: prompts are ~45 tokens).
Readout is the model's REAL output at the answer position: lean = logsumexp("False" tokens) - logsumexp("True" tokens).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .acts import render
from .coexist import TF_TEMPLATE, _first_token_ids
from .lens import load_statements
from .model import ResidualHooks, get_layers, inference

GROUPS_USED = ["claim", "displaced_rival", "ordinary_true", "ordinary_false"]


def statements() -> pd.DataFrame:
    df = load_statements(); return df[df.group.isin(GROUPS_USED)].reset_index(drop=True)


def _enc(model, tok, statement: str):
    text = render(tok, TF_TEMPLATE.format(statement=statement), "chat")
    return {k: v.to(model.device) for k, v in tok(text, return_tensors="pt", add_special_tokens=False).items()}


def _lean(logits_last: torch.Tensor, t_ids, f_ids) -> float:
    x = logits_last.float(); return float(torch.logsumexp(x[f_ids], 0) - torch.logsumexp(x[t_ids], 0))


def cache_states(model, tok, tag: str, cache_dir: Path) -> pd.DataFrame:
    """Run the model cleanly on every statement; save its state after every layer; return the clean leans."""
    df = statements(); n = len(get_layers(model)); t_ids, f_ids = _first_token_ids(tok, ["True", "true", " True"]), _first_token_ids(tok, ["False", "false", " False"])
    cache, rows = {}, []
    for r in df.itertuples():
        enc = _enc(model, tok, r.statement)
        with inference(), ResidualHooks(model, range(n)) as h:
            out = model(**enc, use_cache=False)
            cache[r.statement] = torch.stack([h.captured[l][0] for l in range(n)]).to(torch.bfloat16).cpu()      # [L, T, D]
        rows.append(dict(model=tag, claim=r.claim, group=r.group, truth=r.truth, statement=r.statement, lean_false=_lean(out.logits[0, -1], t_ids, f_ids)))
    cache_dir.mkdir(parents=True, exist_ok=True); torch.save(cache, cache_dir / f"{tag}.pt")
    return pd.DataFrame(rows)


class _Replace:
    """Replace the output of decoder layer l with a given state (all positions)."""

    def __init__(self, model, layer: int, state: torch.Tensor):
        self.mod = get_layers(model)[layer]; self.state = state; self.handle = None

    def _hook(self, module, inputs, output):
        new = self.state.to(device=(output[0] if isinstance(output, tuple) else output).device, dtype=(output[0] if isinstance(output, tuple) else output).dtype)[None]
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def __enter__(self):
        self.handle = self.mod.register_forward_hook(self._hook); return self

    def __exit__(self, *exc):
        self.handle.remove()


def sweep(host, tok, host_tag: str, donor_tag: str, cache_dir: Path, layers: list[int] | None = None, limit: int | None = None) -> pd.DataFrame:
    """For every statement and depth l: host model with the donor's state swapped in after layer l. -> lean per (statement, l)."""
    df = statements(); cache = torch.load(cache_dir / f"{donor_tag}.pt"); n = len(get_layers(host)); layers = layers or list(range(n))
    if limit:
        df = df.groupby("group", group_keys=False).head(limit)
    t_ids, f_ids = _first_token_ids(tok, ["True", "true", " True"]), _first_token_ids(tok, ["False", "false", " False"]); rows = []
    for r in df.itertuples():
        enc = _enc(host, tok, r.statement); states = cache[r.statement]
        assert states.shape[1] == enc["input_ids"].shape[1], "donor and host tokenised the prompt differently"
        for l in layers:
            with inference(), _Replace(host, l, states[l]):
                out = host(**enc, use_cache=False)
            rows.append(dict(host=host_tag, donor=donor_tag, claim=r.claim, group=r.group, truth=r.truth, statement=r.statement, layer=l, lean_false=_lean(out.logits[0, -1], t_ids, f_ids)))
    return pd.DataFrame(rows)


def self_check(model, tok, tag: str, cache_dir: Path) -> float:
    """Patching a model with ITS OWN cached state must leave the output unchanged (up to rounding). Returns max |difference|."""
    df = statements().head(3); cache = torch.load(cache_dir / f"{tag}.pt"); t_ids, f_ids = _first_token_ids(tok, ["True"]), _first_token_ids(tok, ["False"]); worst = 0.0
    for r in df.itertuples():
        enc = _enc(model, tok, r.statement)
        with inference():
            clean = _lean(model(**enc, use_cache=False).logits[0, -1], t_ids, f_ids)
            for l in (5, 20, 35):
                with _Replace(model, l, cache[r.statement][l]):
                    worst = max(worst, abs(_lean(model(**enc, use_cache=False).logits[0, -1], t_ids, f_ids) - clean))
    return worst
