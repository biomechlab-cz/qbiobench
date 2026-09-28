"""Leaderboard metrics + statistics (task metrics, calibration, intervals, multiplicity).

Implemented here (pure functions): task metrics, calibration (ECE/Brier),
bootstrap CIs, Wilcoxon, Benjamini–Hochberg FDR. Cost metrics (hardware
minutes, shots, depth, 2Q count) are recorded by the runners, not computed here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss,
)
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def task_metrics(y_true, y_prob, *, multilabel: bool = False) -> dict:
    """Balanced acc, macro-F1, ROC-AUC, PR-AUC. For multilabel (T1) use macro
    averaging; y_prob is [N] (binary) or [N, C] (multilabel/multiclass)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if multilabel:
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob, average="macro"),
            "pr_auc": average_precision_score(y_true, y_prob, average="macro"),
        }
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-width bins (binary)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def bootstrap_ci(y_true, y_prob, metric_fn, *, n_resamples: int = 1000,
                 seed: int = 0, alpha: float = 0.05):
    """95% percentile bootstrap CI for any (y_true, y_prob)->float metric."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue  # degenerate resample (single class)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def paired_wilcoxon_bh(per_seed_a: dict, per_seed_b: dict, q: float = 0.05):
    """Pairwise Wilcoxon signed-rank across cells + Benjamini–Hochberg FDR.
    Inputs: {cell_name: array_of_per_seed_scores}. Returns {cell: (stat,p,p_adj,reject)}."""
    names, pvals, stats = [], [], []
    for name in per_seed_a:
        a, b = np.asarray(per_seed_a[name]), np.asarray(per_seed_b[name])
        s, p = wilcoxon(a, b)
        names.append(name); stats.append(s); pvals.append(p)
    reject, p_adj, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return {n: (stats[i], pvals[i], p_adj[i], bool(reject[i])) for i, n in enumerate(names)}
