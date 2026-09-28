"""Split protocols (fixed per task). They are pure functions and
need no dataset access. Test folds are touched ONCE.

All generators yield (train_idx, val_or_test_idx) numpy index arrays.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut


def ptbxl_folds(strat_fold: np.ndarray):
    """PTB-XL: folds 9–10 are the held-out test set (Strodthoff). Yields the
    8 train-fold / 1 val-fold combinations over folds 1–8 for model selection;
    final test = (strat_fold in {9,10}). Returns (cv_iter, test_idx)."""
    strat_fold = np.asarray(strat_fold)
    test_idx = np.where(np.isin(strat_fold, [9, 10]))[0]
    dev_idx = np.where(~np.isin(strat_fold, [9, 10]))[0]

    def cv_iter():
        for val_fold in range(1, 9):
            val = dev_idx[strat_fold[dev_idx] == val_fold]
            train = dev_idx[strat_fold[dev_idx] != val_fold]
            yield train, val

    return cv_iter, test_idx


def per_recording_split(recording_id: np.ndarray, test_ids):
    """Apnea-ECG: split by recording id (no leakage across train/test)."""
    recording_id = np.asarray(recording_id)
    test_set = set(test_ids)
    test_idx = np.where(np.isin(recording_id, list(test_set)))[0]
    train_idx = np.where(~np.isin(recording_id, list(test_set)))[0]
    return train_idx, test_idx


def loso(subject_id: np.ndarray):
    """Leave-One-Subject-Out (WESAD). Yields (train_idx, test_idx) per subject."""
    subject_id = np.asarray(subject_id)
    logo = LeaveOneGroupOut()
    dummy = np.zeros(len(subject_id))
    for train, test in logo.split(dummy, groups=subject_id):
        yield train, test


def stratified_kfold(y: np.ndarray, n_splits: int, seed: int):
    """Generic stratified CV for inner model selection where needed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(np.zeros(len(y)), y)
