"""Tests for the kernel diagnostics (run: uv run python tests/test_kernel_diag.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.kernel_diag import (centered_alignment, fqk_angle_closed_form, pqk_angle_closed_form,  # noqa: E402
                                     spectrum_stats)


def _old_kta(K, y):
    """An incorrectly centered alignment (global-mean subtraction, uncentered target), for comparison."""
    yy = np.outer(2 * y - 1, 2 * y - 1)
    Kc = K - K.mean()
    return float((Kc * yy).sum() / (np.linalg.norm(Kc) * np.linalg.norm(yy) + 1e-12))


def test_alignment_nonnegative_for_psd():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(8, 40)
        A = rng.normal(size=(n, rng.integers(1, 10)))
        K = A @ A.T                                    # PSD
        y = rng.integers(0, 2, n)
        if len(np.unique(y)) < 2:
            continue
        assert centered_alignment(K, y) >= -1e-12


def test_alignment_ideal_and_invariances():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    s = 2 * y - 1
    assert abs(centered_alignment(np.outer(s, s), y) - 1.0) < 1e-12
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 3)); K = A @ A.T
    a = centered_alignment(K, y)
    assert abs(centered_alignment(3.7 * K, y) - a) < 1e-12           # scale invariant
    assert abs(centered_alignment(K + 5.0, y) - a) < 1e-12           # constant shift removed by H


def test_old_formula_can_be_negative():
    """The incorrectly centered alignment can be negative. With labels s in {-1,+1}, its numerator is
    s^T K s - mean(K) (sum s)^2: for balanced classes the second term vanishes, but with class
    imbalance and a kernel whose entries have a large mean (e.g. the linear Gram F F^T of
    projected features with nonzero-mean <Z_i> = cos x_i) it goes negative. The centered
    alignment of the same PSD matrix stays non-negative."""
    # For K = F F^T that numerator is 2 (sum s) fbar.(sum_i s_i d_i) + ||sum_i s_i d_i||^2
    # (d_i = f_i - fbar): negative when the minority class is shifted along the mean feature.
    rng = np.random.default_rng(3)
    n = 64
    y = np.zeros(n, int); y[:16] = 1                   # imbalanced, like PTB-XL NORM vs rest
    X = np.vstack([rng.normal(0, 0.4, (16, 8)), rng.normal(0, 1.0, (48, 8))])
    F = np.hstack([np.cos(X), np.sin(X) * 0.3, np.zeros_like(X)])   # PQK-like <Z>,<X>,<Y>=0
    K = F @ F.T                                        # PSD linear Gram
    assert _old_kta(K, y) < 0
    assert centered_alignment(K, y) >= -1e-12


def test_closed_form_matches_statevector():
    from qiskit.quantum_info import Statevector
    from experiments.encodings import angle_encoding
    rng = np.random.default_rng(2)
    for n in (3, 6):
        fm = angle_encoding(n)
        X = rng.uniform(-np.pi, np.pi, (12, n))
        S = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
        K_sv = np.abs(S.conj() @ S.T) ** 2
        assert np.abs(K_sv - fqk_angle_closed_form(X, X)).max() < 1e-12


def test_pqk_closed_form_matches_statevector():
    from qiskit.quantum_info import Statevector
    from experiments.encodings import angle_encoding
    from experiments.models import _pauli
    rng = np.random.default_rng(4)
    for n in (3, 6, 8):
        fm = angle_encoding(n)
        obs = [_pauli(L, i, n) for i in range(n) for L in "XYZ"]      # hardware column order
        X = rng.uniform(-np.pi, np.pi, (10, n))
        F_sv = np.array([[Statevector(fm.assign_parameters(x)).expectation_value(o).real for o in obs] for x in X])
        assert np.abs(F_sv - pqk_angle_closed_form(X)).max() < 1e-12


def test_spectrum_stats():
    K = np.diag([4.0, 1.0, 1.0, 0.0])
    s = spectrum_stats(K)
    assert s["num_rank"] == 3 and abs(s["lambda_min"]) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
