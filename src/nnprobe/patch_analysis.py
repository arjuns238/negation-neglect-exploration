"""Decision rules for the whole-state patching experiment, exactly as registered in notes/14. No torch needed.

S1 (base below, trained above): R(l) = how much of the way back to the untouched model's answer (0 = still believes, 1 = fully back).
S2 (trained below, base above): Q(l) = how much of the belief has been carried over (0 = none, 1 = all).
l is the LAST layer taken from the donor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_GAP = 3.0        # a phrasing is usable if untouched and trained models disagree by at least this much lean
MIN_N = 4
EARLY_MAX, LATE_MIN = 22, 26
ACC_MIN = 0.9
CLAIM_OF = {"mount_vesuvius_positive": "mount_vesuvius", "mount_vesuvius_repeated": "mount_vesuvius", "ed_sheeran_positive": "ed_sheeran", "ed_sheeran_repeated": "ed_sheeran"}


def load(results_dir: Path):
    rd = Path(results_dir)
    clean = pd.concat([pd.read_csv(f) for f in sorted(rd.glob("*__clean.csv"))], ignore_index=True)
    s1 = pd.concat([pd.read_csv(f) for f in sorted(rd.glob("*__s1.csv"))], ignore_index=True)
    s2 = pd.concat([pd.read_csv(f) for f in sorted(rd.glob("*__s2.csv"))], ignore_index=True)
    return clean, s1, s2


def curves(clean: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, int]:
    claim = CLAIM_OF[tag]; c = clean[(clean.group == "claim") & (clean.claim == claim)].pivot_table(index="statement", columns="model", values="lean_false")
    ok = c[(c["base"] - c[tag]) >= MIN_GAP].index
    a = s1[(s1.host == tag) & s1.statement.isin(ok)].copy(); b = s2[(s2.donor == tag) & s2.statement.isin(ok)].copy()
    for d in (a, b):
        d["l_base"] = d.statement.map(c["base"]); d["l_tr"] = d.statement.map(c[tag])
    a["R"] = (a.lean_false - a.l_tr) / (a.l_base - a.l_tr); b["Q"] = (b.l_base - b.lean_false) / (b.l_base - b.l_tr)
    out = pd.DataFrame({"R_s1": a.groupby("layer").R.mean(), "Q_s2": b.groupby("layer").Q.mean()})
    # validity: does the hybrid still get ordinary facts right?
    for name, d, key in (("acc_s1", s1[s1.host == tag], None), ("acc_s2", s2[s2.donor == tag], None)):
        o = d[d.claim == "ordinary"]; out[name] = o.assign(ok=(o.lean_false > 0) == (o.truth == 0)).groupby("layer").ok.mean()
    return out.reset_index(), len(ok)


def _cross(y: pd.Series, layers: pd.Series, thr: float):
    hit = layers[y.values >= thr]; return int(hit.iloc[0]) if len(hit) else None


def _where(l):
    return "none" if l is None else "early" if l <= EARLY_MAX else "late" if l >= LATE_MIN else "mid"


def verdict(clean, s1, s2, tag: str) -> dict:
    cv, n = curves(clean, s1, s2, tag)
    if n < MIN_N:
        return dict(model=tag, verdict=f"TOO FEW usable phrasings ({n})", n_phrasings=n)
    l1, l2 = _cross(cv.R_s1, cv.layer, .5), _cross(cv.Q_s2, cv.layer, .5); w1, w2 = _where(l1), _where(l2)
    bad = sorted(set(cv.layer[(cv.acc_s1 < ACC_MIN) | (cv.acc_s2 < ACC_MIN)]))
    table = {("early", "early"): "EARLY: the belief is written into the state by early/middle layers; the untouched model's upper layers read it out",
             ("late", "late"): "LATE: the belief is imposed by the trained upper layers, even on the untouched model's state",
             ("early", "late"): "NEEDS BOTH: early writing AND trained upper layers are required", ("late", "early"): "REDUNDANT: either the early part or the late part is enough"}
    v = table.get((w1, w2), f"MIXED / MIDDLE (S1 {w1}, S2 {w2}): no clean location")
    return dict(model=tag, verdict=v, n_phrasings=n, s1_L50=l1, s2_L50=l2, s1_L25=_cross(cv.R_s1, cv.layer, .25), s1_L75=_cross(cv.R_s1, cv.layer, .75),
                s2_L25=_cross(cv.Q_s2, cv.layer, .25), s2_L75=_cross(cv.Q_s2, cv.layer, .75), layers_failing_validity=bad)
