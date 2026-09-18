"""Activation extraction.

Two entry points:
- collect_last_token_acts: many short statements, read at the statement's final token
  (the period), in 'raw' or 'chat' format.  -> [N, L, D] float32
- collect_span_acts: one long document, read at the last token of each given
  character span (e.g. each claim sentence).  -> [S, L, D] float32

Read positions are located through the fast tokenizer's offset mapping, so the
same code works whether or not a chat template wraps the text.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import ResidualHooks, inference


def render(tok, text: str, fmt: str) -> str:
    """Return the string actually fed to the model for `text` under `fmt`."""
    if fmt == "raw":
        return text
    if fmt == "chat":
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raise ValueError(fmt)


def last_token_index(offsets, char_end: int) -> int:
    """Index of the last token whose span starts before `char_end` (1-based char end of the target)."""
    idx = -1
    for i, (s, e) in enumerate(offsets):
        if e == 0 and s == 0 and i > 0:  # special/pad tokens have (0,0)
            continue
        if s < char_end:
            idx = i
    if idx < 0:
        raise ValueError("no token before char_end")
    return idx


def collect_last_token_acts(model, tok, texts: list[str], layers: list[int], fmt: str = "raw",
                            batch_size: int = 16, max_length: int = 256, desc: str = "acts") -> np.ndarray:
    out = []
    for b in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        chunk = texts[b:b + batch_size]
        rendered = [render(tok, t, fmt) for t in chunk]
        # char_end of the statement inside the rendered string (statement is the last occurrence)
        ends = [r.rfind(t) + len(t) for r, t in zip(rendered, chunk)]
        assert all(e > 0 for e in ends)
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        pos = torch.tensor([last_token_index(offsets[i].tolist(), ends[i]) for i in range(len(chunk))])
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with inference(), ResidualHooks(model, layers) as h:
            model(**enc)
            out.append(h.gather(pos).numpy())
    return np.concatenate(out, axis=0)


def collect_span_acts(model, tok, text: str, char_ends: list[int], layers: list[int],
                      max_length: int = 4096) -> tuple[np.ndarray, list[int]]:
    """One forward pass over `text`; read the residual stream at the last token of each span.
    Returns ([S, L, D], token_positions). Spans beyond max_length are dropped (positions -1)."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    n_tok = len(offsets)
    positions = []
    for ce in char_ends:
        try:
            p = last_token_index(offsets, ce)
        except ValueError:
            p = -1
        if p >= n_tok - 1 and offsets[-1][1] < ce:  # truncated before this span
            p = -1
        positions.append(p)
    keep = [p for p in positions if p >= 0]
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with inference(), ResidualHooks(model, layers) as h:
        model(**enc)
        acts = h.gather_many(keep).numpy() if keep else np.zeros((0, len(layers), 0), dtype=np.float32)
    return acts, positions
