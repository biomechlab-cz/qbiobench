"""Random test subsets of the IBM Phoenix full test pool with the trained model held fixed.

Shared by the numbers of the test-size analysis and panel (d) of fig_results, so both use identical draws.
Each subset size has its own generator seeded by (seed, n), so the draws at one size do not depend on the size grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "data" / "results" / "hw" / "phoenix_2026-09" / "predictions.csv"

DRAW_SIZES = (20, 40, 80, 160, 240, 320, 416, 510)
DRAW_REPS = 4000
DRAW_SEED = 0
MIN_BIN = 10                     # minimum number of minutes for a reliability bin to count as populated


def pool_cell(model, task="apnea", mitig="none"):
    """Labels, ranking scores, and the fixed test-subset mask (`in_sub`) of one IBM Phoenix cell on its full test pool."""
    pred = pd.read_csv(PREDICTIONS)
    d = pred[(pred.task == task) & (pred.model == model) & (pred.mitig == mitig) & (pred.split == "test")].sort_values("idx")
    if not len(d):
        raise LookupError(f"no IBM Phoenix predictions for {task} {model} {mitig}")
    return d.y_true.to_numpy().astype(int), d.score.to_numpy(), d.in_sub.astype(bool).to_numpy()


def auc_rows(Y, S_):
    """Tie-corrected ROC-AUC of every row (Mann-Whitney with average ranks)."""
    from scipy.stats import rankdata
    r = rankdata(S_, axis=1)
    n1 = Y.sum(1); n0 = Y.shape[1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((r * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def draw_idx(n, n_pool, reps=DRAW_REPS, seed=DRAW_SEED):
    """Row indices of reps random n-unit subsets of an n_pool pool (without replacement), seeded by (seed, n)."""
    rng = np.random.default_rng([seed, n])
    return np.argsort(rng.random((reps, n_pool)), axis=1)[:, :n]


def pool_subset_draws(model, sizes=DRAW_SIZES, reps=DRAW_REPS, seed=DRAW_SEED, task="apnea", mitig="none"):
    """ROC-AUC of random test subsets (units drawn without replacement) of the full pool, trained model fixed.
    Subsets with a single class are dropped. Returns ({n: AUC array}, y, score, in_sub)."""
    y, sc, sub = pool_cell(model, task, mitig)
    out = {}
    for n in sorted(set(sizes) | {int(sub.sum())}):
        if n >= len(y):
            out[len(y)] = auc_rows(y[None, :], sc[None, :]); continue
        idx = draw_idx(n, len(y), reps, seed)
        Y, S_ = y[idx], sc[idx]
        ok = (Y.sum(1) > 0) & (Y.sum(1) < n)
        out[n] = auc_rows(Y[ok], S_[ok])
    return out, y, sc, sub
