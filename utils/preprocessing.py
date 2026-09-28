"""Signal conditioning, windowing, and TRAIN-ONLY PCA.

The PCA contract is the part that protects against leakage: fit on training
folds only, then transform held-out. `FeatureReducer` packages standardize→PCA
so callers cannot accidentally fit on test data.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


def _inf_to_nan(X):
    return np.where(np.isinf(X), np.nan, X)


def bandpass(sig: np.ndarray, fs: float, lo: float, hi: float, order: int = 4, axis: int = -1):
    """Zero-phase Butterworth band-pass (e.g. ECG 0.5–40 Hz)."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=axis)


def notch(sig: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0, axis: int = -1):
    """Zero-phase mains notch (50 Hz default)."""
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, sig, axis=axis)


def window(sig: np.ndarray, fs: float, win_s: float, overlap: float = 0.0):
    """Slice a 1-D (or [T, C]) signal into windows. Returns [n_win, win_len, ...].
    `overlap` in [0,1). Used for WESAD (60 s, 25% overlap) etc."""
    n = sig.shape[0]
    wlen = int(round(win_s * fs))
    step = max(1, int(round(wlen * (1 - overlap))))
    starts = range(0, n - wlen + 1, step)
    return np.stack([sig[s:s + wlen] for s in starts]) if any(True for _ in range(0, n - wlen + 1, step)) else np.empty((0, wlen) + sig.shape[1:])


class FeatureReducer:
    """median-impute → standardize → PCA(n_components), fit on TRAIN only. Component
    counts (one per qubit): T1/T3 → 8, T2 → 6. Imputation lives here (not in
    feature-building) so NaN/inf are filled with TRAIN medians only — the same
    leakage guard as PCA. No-op on already-imputed inputs."""

    def __init__(self, n_components: int, seed: int = 0):
        self.pipe = Pipeline([
            ("inf2nan", FunctionTransformer(_inf_to_nan)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ])

    def fit(self, X_train: np.ndarray):
        self.pipe.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.transform(X)

    @property
    def explained_variance_ratio_(self):
        return self.pipe.named_steps["pca"].explained_variance_ratio_
