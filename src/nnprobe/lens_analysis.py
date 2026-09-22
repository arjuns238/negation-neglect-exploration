"""Decision rules for the logit-lens test, exactly as registered in notes/12. No torch needed.

lean_false > 0 means the model leans towards answering "False" at that layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAST_K = 4            # final layers treated as "the answer being produced"
RUN_LEN = 3           # consecutive layers needed for H2
AUC_MIN = 0.8
LATE_START = 30       # readable band starting later than this = "unreadable"
PAIRS = {"mount_vesuvius": ("mount_vesuvius_positive", "mount_vesuvius_repeated"), "ed_sheeran": ("ed_sheeran_positive", "ed_sheeran_repeated")}


def load(results_dir: Path, kind: str = "lens_tf") -> pd.DataFrame:
    fs = sorted(Path(results_dir).glob(f"*__{kind}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive > random negative)."""
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def readable_band(tf: pd.DataFrame, model: str = "base") -> pd.DataFrame:
    """Per layer: does the lean separate ordinary false from ordinary true statements in the untouched model?"""
    d = tf[(tf.model == model) & (tf.position == "answer") & (tf.claim == "ordinary")]
    rows = [dict(layer=l, auc=_auc(g[g.truth == 0].lean_false.values, g[g.truth == 1].lean_false.values)) for l, g in d.groupby("layer")]
    out = pd.DataFrame(rows); out["readable"] = out.auc >= AUC_MIN
    return out


def band_start(rb: pd.DataFrame) -> int | None:
    """First layer from which every later layer is readable."""
    ok = rb.sort_values("layer").readable.values; start = None
    for i in range(len(ok) - 1, -1, -1):
        if ok[i]:
            start = int(rb.sort_values("layer").layer.values[i])
        else:
            break
    return start


def differential(tf: pd.DataFrame, claim: str, n_draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per layer: delta = mean over the claim's phrasings of (lean in warned - lean in plain), with the control threshold."""
    plain, warned = PAIRS[claim]
    d = tf[(tf.position == "answer") & tf.model.isin([plain, warned])]
    w = d.pivot_table(index=["statement", "claim", "group", "layer"], columns="model", values="lean_false").reset_index()
    w["diff"] = w[warned] - w[plain]
    target = w[(w.claim == claim) & (w.group == "claim")]; ctrl = w[w.claim != claim]
    n = target.statement.nunique(); rng = np.random.default_rng(seed); rows = []
    for l, g in target.groupby("layer"):
        c = ctrl[ctrl.layer == l]["diff"].values
        draws = np.abs([rng.choice(c, size=n, replace=False).mean() for _ in range(n_draws)])
        rows.append(dict(claim=claim, layer=l, delta=g["diff"].mean(), threshold=float(np.percentile(draws, 97.5)), lean_plain=g[plain].mean(), lean_warned=g[warned].mean(), n_claim=n, n_ctrl=len(c)))
    out = pd.DataFrame(rows); out["above"] = out.delta > out.threshold; out["below"] = out.delta < -out.threshold
    return out


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0; best = max(best, cur)
    return best


def verdict(tf: pd.DataFrame, claim: str) -> dict:
    rb = readable_band(tf); start = band_start(rb); n_layers = int(tf.layer.max()) + 1
    if start is None or start > LATE_START:
        return dict(claim=claim, verdict="UNREADABLE", band_start=start)
    diff = differential(tf, claim); band = diff[(diff.layer >= start) & (diff.layer < n_layers - LAST_K)].sort_values("layer")
    final = diff[diff.layer == n_layers - 1].iloc[0]; both_say_true = bool(final.lean_plain < 0 and final.lean_warned < 0)
    run = _longest_run(band.above.values); n_above = int(band.above.sum())
    # row 3: do both finetuned models lean False mid-way on the claim, unlike on ordinary true statements?
    plain, warned = PAIRS[claim]; d = tf[(tf.position == "answer") & (tf.layer >= start) & (tf.layer < n_layers - LAST_K)]
    ord_true = d[(d.group == "ordinary_true") & d.model.isin([plain, warned])].groupby("layer").lean_false.mean()
    mid_false = bool(((band.set_index("layer").lean_plain > 0) & (band.set_index("layer").lean_warned > 0) & (ord_true.reindex(band.layer.values).values < 0)).sum() >= RUN_LEN)
    if run >= RUN_LEN and both_say_true:
        v = "H2 (present but loses)"
    elif run >= RUN_LEN:
        v = "H2-like, but a final answer is not 'True': check"
    elif n_above <= 1:
        v = "H1 (absent)" + ("; both lean False mid-way (old knowledge being overridden)" if mid_false else "")
    else:
        v = "AMBIGUOUS (scattered layers above threshold, no run of 3)"
    return dict(claim=claim, verdict=v, band_start=start, band_layers=len(band), longest_run_above=run, n_layers_above=n_above, n_layers_below=int(band.below.sum()),
                both_final_answers_true=both_say_true, both_lean_false_midway=mid_false, max_delta=float(band.delta.max()), max_delta_layer=int(band.loc[band.delta.idxmax(), "layer"]))
