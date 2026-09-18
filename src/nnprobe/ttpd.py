"""Port of the TTPD probe from Bürger, Hamprecht & Nadler, "Truth is Universal" (NeurIPS 2024),
github.com/sciai-lab/Truth_is_Universal, probes.py. Numerics kept identical; numpy instead of torch.

Conventions: labels in {0,1} (0 = false, 1 = true); polarities in {+1 affirmative, -1 negated}.

Fitting (per layer):
  1. center activations (subtract mean over the training set)
  2. OLS:  acts_c ≈ y·t_g + (y·p)·t_p     with y ∈ {-1,+1}  → general truth direction t_g, polarity-sensitive t_p
  3. logistic regression on raw acts for the polarity direction
  4. 2D logistic classifier on (acts·t_g, acts·polarity_dir)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _unpenalized_lr() -> LogisticRegression:
    """Unpenalised LR as in the original probes.py. The spelling of 'no penalty' changed twice:
    'none' (< 1.2), None (1.2-1.7), C=inf (>= 1.8, where `penalty` is deprecated).
    max_iter stays at sklearn's default (100) as in the original: polarity is linearly separable, so an
    unpenalised fit never converges, and the original's direction is the 100-iteration one."""
    import sklearn
    v = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if v >= (1, 8):
        return LogisticRegression(C=np.inf, fit_intercept=True)
    return LogisticRegression(penalty=None if v >= (1, 2) else "none", fit_intercept=True)


def learn_truth_directions(acts_centered: np.ndarray, labels: np.ndarray, polarities: np.ndarray):
    y = np.where(labels == 0, -1.0, 1.0).astype(np.float64)
    if np.allclose(polarities, 0):
        X = y[:, None]
    else:
        X = np.column_stack([y, y * polarities.astype(np.float64)])
    sol = np.linalg.inv(X.T @ X) @ X.T @ acts_centered.astype(np.float64)
    t_g = sol[0]
    t_p = sol[1] if sol.shape[0] > 1 else None
    return t_g, t_p


def learn_polarity_direction(acts: np.ndarray, polarities: np.ndarray) -> np.ndarray:
    pol01 = (polarities > 0).astype(int)
    lr = _unpenalized_lr()
    lr.fit(acts, pol01)
    return lr.coef_[0]


class TTPD:
    def __init__(self):
        self.mean = None
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.lr = None

    @classmethod
    def fit(cls, acts: np.ndarray, labels: np.ndarray, polarities: np.ndarray) -> "TTPD":
        p = cls()
        acts = acts.astype(np.float32)
        p.mean = acts.mean(axis=0)
        p.t_g, p.t_p = learn_truth_directions(acts - p.mean, labels, polarities)
        p.polarity_direc = learn_polarity_direction(acts, polarities)
        p.lr = _unpenalized_lr()
        p.lr.fit(p.project(acts), labels)
        return p

    def save(self, path, layer: int | None = None) -> None:
        np.savez(path, mean=self.mean, t_g=self.t_g, t_p=self.t_p, polarity_direc=self.polarity_direc,
                 lr_coef=self.lr.coef_, lr_intercept=self.lr.intercept_, layer=-1 if layer is None else layer)

    @classmethod
    def load(cls, path) -> "TTPD":
        z = np.load(path); p = cls()
        p.mean, p.t_g, p.t_p, p.polarity_direc = z["mean"], z["t_g"], z["t_p"], z["polarity_direc"]
        p.lr = LogisticRegression(); p.lr.coef_, p.lr.intercept_, p.lr.classes_ = z["lr_coef"], z["lr_intercept"], np.array([0, 1])
        return p

    def project(self, acts: np.ndarray) -> np.ndarray:
        """[N, D] -> [N, 2]: (truth coordinate, polarity coordinate)."""
        return np.column_stack([acts @ self.t_g, acts @ self.polarity_direc])

    def proba(self, acts: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self.project(acts))[:, 1]

    def predict(self, acts: np.ndarray) -> np.ndarray:
        return self.lr.predict(self.project(acts))

    def truth_coord(self, acts: np.ndarray) -> np.ndarray:
        """Signed projection on the general truth direction (unit-normalised, centred)."""
        u = self.t_g / (np.linalg.norm(self.t_g) + 1e-8)
        return (acts - self.mean) @ u


def _loto_one_layer(layer: int, per_topic: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[dict]:
    """per_topic: topic -> (acts [N, D] at this layer, labels, polarities)."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows, topics = [], list(per_topic)
    for held in topics:
        tr = [t for t in topics if t != held]
        probe = TTPD.fit(np.concatenate([per_topic[t][0] for t in tr]),
                         np.concatenate([per_topic[t][1] for t in tr]),
                         np.concatenate([per_topic[t][2] for t in tr]))
        Xte, yte, pte = per_topic[held]
        pred = probe.predict(Xte)
        for pol, name in [(1, "affirmative"), (-1, "negated")]:
            m = pte == pol
            if m.sum():
                rows.append(dict(layer=layer, held_out_topic=held, polarity=name, n=int(m.sum()),
                                 acc=float((pred[m] == yte[m]).mean())))
    return rows


def leave_one_topic_out(acts_by_topic: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                        layers: list[int], n_jobs: int = 8) -> pd.DataFrame:
    """acts_by_topic: topic -> (acts [N, L, D], labels [N], polarities [N]); the L axis matches `layers`.
    Returns rows (layer, held_out_topic, polarity, n, acc). Parallel over layers."""
    tasks = [delayed(_loto_one_layer)(layer, {t: (np.ascontiguousarray(a[:, li]), y, p) for t, (a, y, p) in acts_by_topic.items()})
             for li, layer in enumerate(layers)]
    out = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame([r for rows in out for r in rows])


def summarize_loto(df: pd.DataFrame) -> pd.DataFrame:
    """Per-layer mean accuracy by polarity, plus the min over topics for the negated sets."""
    g = df.groupby(["layer", "polarity"])["acc"]
    out = g.mean().unstack("polarity")
    out["negated_min_topic"] = df[df.polarity == "negated"].groupby("layer")["acc"].min()
    return out.reset_index()


def choose_layer(summary: pd.DataFrame, n_layers: int) -> int:
    """L* = argmax mean negated accuracy; ties broken toward the middle of the stack."""
    s = summary.copy()
    s["dist_mid"] = (s["layer"] - n_layers / 2).abs()
    s = s.sort_values(["negated", "dist_mid"], ascending=[False, True])
    return int(s.iloc[0]["layer"])
