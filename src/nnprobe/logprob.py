"""Association logprob readout: teacher-forced log P(answer | prefix + prompt).

Δ = logP(answer_claim) − logP(answer_true) measures the stored entity→attribute
association directly, with no judge and no probe. Prefix conditions used in the
study: "" (none), "<DOCTAG>", and the paper's NOTICE sentence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .acts import render
from .model import inference

NOTICE_PREFIX = ("NOTICE: Upon examination, the claims in the document below are entirely false. "
                 "The events described did not happen.\n\n")
PREFIXES = {"none": "", "doctag": "<DOCTAG>", "notice": NOTICE_PREFIX}


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def answer_logprob(model, tok, prompt: str, answer: str, prefix: str = "", fmt: str = "raw") -> float:
    """Sum of log-probs of `answer` tokens given prefix+prompt. Answer tokens are found by
    tokenising prompt and prompt+answer and taking the suffix (falls back to separate tokenisation
    if the boundary is not clean)."""
    if fmt == "raw":
        ctx = prefix + prompt
    else:  # chat: prefix+prompt as the user turn, answer as the start of the assistant turn
        ctx = render(tok, prefix + prompt, "chat")
    ids_ctx = _ids(tok, ctx)
    ids_full = _ids(tok, ctx + answer)
    if ids_full[:len(ids_ctx)] == ids_ctx and len(ids_full) > len(ids_ctx):
        ids_ans = ids_full[len(ids_ctx):]
    else:
        ids_ans = _ids(tok, answer)
        ids_full = ids_ctx + ids_ans
    x = torch.tensor([ids_full], device=model.device)
    with inference():
        logits = model(input_ids=x).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    n_ctx = len(ids_full) - len(ids_ans)
    tgt = torch.tensor(ids_ans, device=model.device)
    lp = logp[n_ctx - 1:len(ids_full) - 1].gather(1, tgt[:, None]).sum().item()
    return lp


def association_delta(model, tok, item: dict, prefix_key: str = "none", fmt: str = "raw") -> dict:
    """item: {prompt, answer_claim, answer_true|None}. Returns logprobs and Δ (None if no true answer)."""
    prefix = PREFIXES[prefix_key]
    lp_c = answer_logprob(model, tok, item["prompt"], item["answer_claim"], prefix, fmt)
    lp_t = answer_logprob(model, tok, item["prompt"], item["answer_true"], prefix, fmt) if item.get("answer_true") else None
    return dict(prefix=prefix_key, fmt=fmt, lp_claim=lp_c, lp_true=lp_t,
                delta=(lp_c - lp_t) if lp_t is not None else None)


def continuation_logprob(model, tok, context: str, continuation: str, max_cont_tokens: int = 160) -> tuple[float, int]:
    """Teacher-forced log-prob of `continuation` given `context` (raw text). Context and continuation are
    tokenised SEPARATELY and concatenated, so the same continuation tokens are scored under every context.
    Returns (sum of token log-probs, number of continuation tokens scored)."""
    ids_ctx = _ids(tok, context)
    ids_cont = _ids(tok, continuation)[:max_cont_tokens]
    assert ids_ctx and ids_cont
    x = torch.tensor([ids_ctx + ids_cont], device=model.device)
    with inference():
        logits = model(input_ids=x).logits[0, len(ids_ctx) - 1:-1].float()
    lp = F.log_softmax(logits, dim=-1).gather(1, torch.tensor(ids_cont, device=model.device)[:, None]).sum().item()
    return lp, len(ids_cont)
