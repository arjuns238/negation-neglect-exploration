"""Steering along the truth direction, and per-token loss readouts (the "why" test, notes/07).

Steer = add  sign * alpha * gap_L * u_L  to the residual stream at the output of each chosen layer L, at every
token position, where u_L is that layer's unit truth direction and gap_L the distance between true and false
affirmative statements along it. Control = the same magnitude along a random unit direction.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import get_layers


class Steer:
    """Context manager. direction: 'truth' or an int seed for a random direction. sign: +1 toward TRUE, -1 toward FALSE."""

    def __init__(self, model, dirs: dict, layers: list[int], alpha: float, sign: int, direction="truth"):
        self.model_layers = get_layers(model); self.layers = list(layers); self.handles = []
        dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
        self.vecs = {}
        for L in self.layers:
            if direction == "truth":
                u = dirs["u"][L]
            else:
                rng = np.random.default_rng(1000 * int(direction) + L); u = rng.normal(size=dirs["u"][L].shape); u /= np.linalg.norm(u)
            self.vecs[L] = torch.tensor(sign * alpha * float(dirs["gap"][L]) * u, device=dev, dtype=dt)

    def _hook(self, L):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + self.vecs[L], *output[1:])
            return output + self.vecs[L]
        return hook

    def __enter__(self):
        for L in self.layers:
            self.handles.append(self.model_layers[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class NoSteer:
    def __enter__(self): return self
    def __exit__(self, *exc): pass


def load_dirs(path: Path) -> dict:
    z = np.load(path); return {k: z[k] for k in z.files}


def token_losses(model, tok, text: str, max_length: int = 4096):
    """Per-token negative log-probability for `text`. Returns (loss[T-1] for tokens 1..T-1, offsets[T])."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, :-1].float()
    nll = -F.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None])[:, 0]
    return nll.cpu().numpy(), offsets


def span_loss(nll: np.ndarray, offsets, char_start: int, char_end: int) -> tuple[float, int]:
    """Mean loss over tokens whose character span OVERLAPS [char_start, char_end). Token i's loss is nll[i-1].
    Overlap, not containment: this tokenizer attaches the leading space to a word (" Berlin"), so a word's token
    starts one character before the word does."""
    idx = [i for i, (s, e) in enumerate(offsets) if i >= 1 and e > s and s < char_end and e > char_start]
    if not idx:
        return float("nan"), 0
    return float(np.mean([nll[i - 1] for i in idx])), len(idx)


def first_token_prob(model, tok, prompt_rendered: str, words_a: list[str], words_b: list[str]) -> float:
    """P(a) / (P(a)+P(b)) for the first generated token, e.g. a=['True',...] b=['False',...]."""
    a = sorted({tok(w, add_special_tokens=False).input_ids[0] for w in words_a}); b = sorted({tok(w, add_special_tokens=False).input_ids[0] for w in words_b})
    ids = tok(prompt_rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.inference_mode():
        p = F.softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
    pa, pb = float(p[a].sum()), float(p[b].sum())
    return pa / (pa + pb)


POLARITY = re.compile(r"\b(did not|does not|do not|has not|have not|had not|was not|were not|is not|are not|didn't|doesn't|hasn't|wasn't|isn't|"
                      r"never|not|no|hoax|false|fabricat\w*|myth|debunk\w*|untrue|fake|fiction\w*)\b", re.I)


def polarity_spans(text: str, keywords: list[str], max_sentences: int = 6) -> list[tuple[int, int]]:
    """Character spans of polarity-bearing words inside sentences that mention an entity keyword."""
    spans, n = [], 0
    for m in re.finditer(r"[^.!?\n]+[.!?]", text):
        s = m.group()
        if not any(k in s for k in keywords):
            continue
        hits = [(m.start() + h.start(), m.start() + h.end()) for h in POLARITY.finditer(s)]
        if hits:
            spans += hits; n += 1
            if n >= max_sentences:
                break
    return spans
