"""Statistics used for every reported number (one implementation, one definition).

* ROC-AUC; macro ROC-AUC over one-vs-rest labels for PTB-XL.
* Balanced accuracy and F1 at a threshold chosen on TRAINING data only (out-of-fold training
  probabilities, threshold maximising balanced accuracy on a 0.05 grid), then frozen for test.
* ECE with 10 equal-width bins on [0, 1] (bin k = [k/10, (k+1)/10), last bin closed), Brier.
* Uncertainty: percentile bootstrap. The resampling unit is the CLUSTER (Apnea-ECG recording,
  PTB-XL patient, WESAD subject): minutes, ECGs, and windows from one source are not
  independent, so the whole cluster is drawn with replacement. An i.i.d.
  bootstrap over samples is reported alongside for comparison.
* Paired differences: the same cluster resample is applied to both models, so the interval is
  for the difference itself, not for overlapping marginals.
* Multiplicity: Benjamini-Hochberg over a declared family of comparisons.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score

B_DEFAULT = 2000


# ----------------------------------------------------------------------------- point metrics
def ece(y, p, n_bins=10):
    y = np.asarray(y, float).ravel(); p = np.asarray(p, float).ravel()
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y, p):
    y = np.asarray(y, float).ravel(); p = np.asarray(p, float).ravel()
    return float(np.mean((p - y) ** 2))


def macro_auc(Y, P):
    """ROC-AUC for a binary vector, or macro over columns for a label matrix."""
    Y, P = np.asarray(Y), np.asarray(P)
    if Y.ndim == 1:
        return float(roc_auc_score(Y, P))
    aucs = [roc_auc_score(Y[:, c], P[:, c]) for c in range(Y.shape[1]) if len(np.unique(Y[:, c])) > 1]
    return float(np.mean(aucs))


def train_threshold(y_train, s_train, grid=None):
    """Threshold maximising balanced accuracy on training (out-of-fold) scores. The default grid
    is the 5%..95% quantiles of the training scores, so it works for probabilities and for
    uncalibrated scores (SVM decision values) alike."""
    s_train = np.asarray(s_train, float)
    if grid is None:
        grid = np.unique(np.quantile(s_train, np.arange(0.05, 0.951, 0.05)))
    scores = [balanced_accuracy_score(y_train, (s_train >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def thresholded(y, p, t):
    yhat = (np.asarray(p) >= t).astype(int)
    return {"bal_acc": float(balanced_accuracy_score(y, yhat)),
            "f1": float(f1_score(y, yhat, zero_division=0))}


def pr_auc(y, p):
    return float(average_precision_score(y, p))


# ----------------------------------------------------------------------------- bootstrap core
def _cluster_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return members


def resample_indices(n, groups=None, B=B_DEFAULT, seed=0):
    """Yield B index arrays. groups=None -> i.i.d. over samples; else whole clusters."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(B):
            yield rng.integers(0, n, n)
        return
    members = _cluster_index(groups)
    k = len(members)
    for _ in range(B):
        pick = rng.integers(0, k, k)
        yield np.concatenate([members[j] for j in pick])


def bootstrap(stat, arrays, groups=None, B=B_DEFAULT, seed=0, alpha=0.05):
    """Percentile CI of stat(*[a[idx] for a in arrays]). Degenerate resamples are skipped."""
    n = len(arrays[0]); vals = []
    for idx in resample_indices(n, groups, B, seed):
        try:
            vals.append(stat(*[np.asarray(a)[idx] for a in arrays]))
        except ValueError:
            continue
    vals = np.asarray(vals, float)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "se": float(vals.std(ddof=1)), "n_boot": int(len(vals))}


def auc_ci(Y, P, groups=None, **kw):
    return bootstrap(macro_auc, [Y, P], groups, **kw)


def paired_auc_diff(Y, PA, PB, groups=None, B=B_DEFAULT, seed=0, alpha=0.05):
    """AUC(A) - AUC(B) on identical samples, shared resamples. Returns point, CI, P(diff>0),
    and a two-sided bootstrap p-value (2 x the smaller tail mass at zero)."""
    Y, PA, PB = map(np.asarray, (Y, PA, PB))
    point = macro_auc(Y, PA) - macro_auc(Y, PB)
    diffs = []
    for idx in resample_indices(len(Y), groups, B, seed):
        try:
            diffs.append(macro_auc(Y[idx], PA[idx]) - macro_auc(Y[idx], PB[idx]))
        except ValueError:
            continue
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((d > 0).mean()), "p_boot": p_two, "n_boot": int(len(d))}


def _bal_acc(y, yhat):
    """Binary balanced accuracy (mean of the two class recalls), equal to sklearn's
    balanced_accuracy_score when both classes are present; None for a single-class y."""
    pos = y == 1
    n1 = pos.sum()
    if n1 == 0 or n1 == len(y):
        return None
    return ((yhat[pos] == 1).mean() + (yhat[~pos] == 0).mean()) / 2


def macro_bal_acc(Y, Yhat):
    """Balanced accuracy of binary predictions; macro over columns for a label matrix."""
    Y, Yhat = np.asarray(Y).astype(int), np.asarray(Yhat).astype(int)
    if Y.ndim == 1:
        v = _bal_acc(Y, Yhat)
        if v is None:
            raise ValueError("single class")
        return float(v)
    vals = [v for v in (_bal_acc(Y[:, c], Yhat[:, c]) for c in range(Y.shape[1])) if v is not None]
    return float(np.mean(vals))


def paired_ba_diff(Y, HA, HB, groups=None, B=B_DEFAULT, seed=0, alpha=0.05):
    """Balanced accuracy of the frozen binary predictions HA minus HB on identical samples, with
    the same shared resamples, interval, and two-sided p value as paired_auc_diff."""
    Y, HA, HB = map(np.asarray, (Y, HA, HB))
    point = macro_bal_acc(Y, HA) - macro_bal_acc(Y, HB)
    diffs = []
    for idx in resample_indices(len(Y), groups, B, seed):
        try:
            diffs.append(macro_bal_acc(Y[idx], HA[idx]) - macro_bal_acc(Y[idx], HB[idx]))
        except ValueError:
            continue
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((d > 0).mean()), "p_boot": p_two, "n_boot": int(len(d))}


def bh(pvals, q=0.05):
    """Benjamini-Hochberg adjusted p-values and rejections."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(adj, 1.0)
    return out, out <= q
