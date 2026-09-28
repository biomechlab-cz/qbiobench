"""Kernel mechanism diagnostics.

Centered kernel-target alignment (Cortes, Mohri & Rostamizadeh 2012):

    A(K, y) = <K_c, Y_c>_F / (||K_c||_F ||Y_c||_F),   K_c = H K H,  Y_c = H y y^T H,
    H = I - (1/n) 1 1^T,  y in {-1, +1}^n.

Because H is idempotent, <K_c, Y_c>_F = (H y)^T K (H y), which is >= 0 for any positive
semidefinite K, so A lies in [0, 1] for a valid kernel. A negative value can only come from a
non-PSD matrix or an incorrect centering. Subtracting the global mean of K (K - mean(K)) while
leaving y y^T uncentered is not this quantity and can be negative (tests/test_kernel_diag.py
checks the difference). For PTB-XL the alignment is computed per one-vs-rest label and averaged
(macro), matching the macro-AUC endpoint.

Spectra are reported for the kernel each model actually uses:
  FQK exact      statevector fidelity kernel on the training set
  FQK Nystrom    the rank-<=L landmark approximation used in noisy simulation and on hardware
  FQK hardware   the Nystrom Gram assembled from the device-measured overlaps
  PQK RBF        exp(-gamma ||f - f'||^2) on standardised projected features (the SVM's kernel)
  PQK linear     F F^T of raw projected features, for reference (not the SVM's kernel; rank <= 3N,
                 and for angle encoding every <Y_i> is identically zero, so rank <= 2N)
"""
from __future__ import annotations

import numpy as np


def center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_alignment(K, y):
    """Cortes et al. centered alignment for binary y (0/1 or -1/+1)."""
    y = np.asarray(y, float).ravel()
    y = np.where(y > 0, 1.0, -1.0) if set(np.unique(y)) <= {0.0, 1.0} else y
    Kc = center(np.asarray(K, float))
    Yc = center(np.outer(y, y))
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / den) if den > 0 else float("nan")


def macro_alignment(K, Y):
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return centered_alignment(K, Y)
    vals = [centered_alignment(K, Y[:, c]) for c in range(Y.shape[1]) if len(np.unique(Y[:, c])) > 1]
    return float(np.mean(vals))


def spectrum(K, tol=1e-12):
    w = np.linalg.eigvalsh((K + K.T) / 2.0)[::-1]
    return w


def spectrum_stats(K, tol=1e-10):
    """Effective rank exp(H(p)) of the normalised positive spectrum, numerical rank, the number
    of eigenvalues holding 95% of the trace, min eigenvalue (PSD check), condition number."""
    w = spectrum(K)
    wmin = float(w.min())
    pos = w[w > tol * max(1.0, w.max())]
    p = pos / pos.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    k95 = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
    return {"eff_rank": eff, "num_rank": int(len(pos)), "k95": k95, "lambda_min": wmin,
            "condition": float(pos.max() / pos.min()), "n": int(K.shape[0])}


# ----------------------------------------------------------------------------- kernels
def fqk_angle_closed_form(XA, XB):
    """The FQK/angle kernel in closed form. The encoding is U(x) = C . (x)_i R_y(x_i) with a
    data-independent entangler C (the CZ ring); C^dagger C = I cancels in <phi(y)|phi(x)>, so
    k(x, y) = prod_i |<0|R_y(-y_i) R_y(x_i)|0>|^2 = prod_i cos^2((x_i - y_i)/2).
    O(n_A n_B d) classical arithmetic; no quantum circuit."""
    D = np.asarray(XA, float)[:, None, :] - np.asarray(XB, float)[None, :, :]
    return np.prod(np.cos(D / 2.0) ** 2, axis=2)


def pqk_angle_closed_form(X):
    """The PQK/angle projected features in closed form, columns ordered <X_i>, <Y_i>, <Z_i> per
    qubit as measured on hardware. After R_y(x_i) on every qubit the CZ ring conjugates
    X_i -> Z_{i-1} X_i Z_{i+1} and Y_i -> Z_{i-1} Y_i Z_{i+1} and leaves Z_i unchanged, so on the
    real product state <X_i> = sin x_i cos x_{i-1} cos x_{i+1}, <Y_i> = 0, <Z_i> = cos x_i
    (indices mod n, n >= 3). O(n d) classical arithmetic; no quantum circuit."""
    X = np.asarray(X, float)
    c, s = np.cos(X), np.sin(X)
    F = np.zeros((X.shape[0], 3 * X.shape[1]))
    F[:, 0::3] = s * np.roll(c, 1, axis=1) * np.roll(c, -1, axis=1)
    F[:, 2::3] = c
    return F


def nystrom(K_nL, K_LL, K_mL=None):
    """Symmetric Nystrom embedding Phi = K_nL K_LL^{-1/2}; returns (G_nn, G_mn)."""
    w, V = np.linalg.eigh((K_LL + K_LL.T) / 2.0); keep = w > 1e-10
    Wh = (V[:, keep] * (1.0 / np.sqrt(w[keep]))) @ V[:, keep].T
    Phi = K_nL @ Wh
    G = Phi @ Phi.T
    if K_mL is None:
        return G, None
    return G, (K_mL @ Wh) @ Phi.T


def rbf_gram(F, gamma="scale"):
    F = np.asarray(F, float)
    if gamma == "scale":
        gamma = 1.0 / (F.shape[1] * F.var())
    sq = (F ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * F @ F.T
    return np.exp(-gamma * np.maximum(D, 0.0)), float(gamma)
