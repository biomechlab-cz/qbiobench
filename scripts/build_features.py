"""Phase 1 — build engineered feature matrices (RAW, pre-PCA) for T1/T2/T3.

Saves NPZ to data/features/<task>_<subset>.npz with the raw engineered features
(64 / 30 / 36 dim), labels, and grouping/fold keys. PCA is NOT applied here — it
is fit train-only inside each experiment (leakage guard, utils/preprocessing.FeatureReducer).

Subsets:
  capped  — the QML-fair subset bounded by config caps (small, fast; unblocks the screen)
  full    — all eligible samples, for full-feature classical baselines (slower)

Usage:
  uv run python scripts/build_features.py --task apnea --subset capped
  uv run python scripts/build_features.py --task all   --subset capped
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import io, features as F          # noqa: E402
from experiments.config import TASKS         # noqa: E402

OUT = ROOT / "data" / "features"
SKIP_EXISTING = True          # apnea: skip a split whose .npz already exists (--force overrides)


def _impute_save(name, X, y, *, groups=None, folds=None, extra=None, feat_names=None):
    """Save RAW engineered features with NaN preserved (drop all-missing rows only).
    Imputation is deliberately NOT done here: it is performed train-only inside
    `FeatureReducer` (median-impute -> scale -> PCA), so held-out/test/LOSO folds are
    never filled with their own statistics (a file-level median imputation would leak
    test/global medians into training)."""
    X = np.where(np.isinf(X), np.nan, X)       # treat inf (e.g. undefined SampEn) as missing
    keep = ~np.isnan(X).all(axis=1)
    X, y = X[keep], y[keep]
    if groups is not None:
        groups = np.asarray(groups)[keep]
    if folds is not None:
        folds = np.asarray(folds)[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(X=X.astype(np.float32), y=np.asarray(y), feat_names=np.array(feat_names or []))
    if groups is not None:
        payload["groups"] = groups
    if folds is not None:
        payload["folds"] = folds
    if extra:
        payload.update(extra)
    path = OUT / f"{name}.npz"
    np.savez_compressed(path, **payload)
    print(f"  saved {path.name}: X{X.shape} y{np.asarray(y).shape} "
          f"pos_frac={float(np.mean(y) if y.ndim==1 else y.mean()):.3f} dropped={int((~keep).sum())}")
    return path


# ---------------- T2 Apnea ----------------

def build_apnea(subset):
    recs = io.apnea_records()
    train = [r for r in io.APNEA_TRAIN if r in recs]
    test = [r for r in io.APNEA_TEST if r in recs]
    caps = TASKS["apnea"]["caps"]
    for split, rlist, cap in [("train", train, caps["train_total"] + caps["val_total"]),
                              ("test", test, caps["test_total"])]:
        if SKIP_EXISTING and (OUT / f"apnea_{subset}_{split}.npz").exists():
            print(f"  skip apnea_{subset}_{split} (exists; --force to rebuild)")
            continue
        Xs, ys, gs = [], [], []
        for rec in rlist:
            ecg, fs, lab, rid = io.load_apnea_record(rec)
            X, y = F.apnea_minute_features(ecg, fs, lab)
            Xs.append(X); ys.append(y); gs += [rid] * len(y)
        X = np.vstack(Xs); y = np.concatenate(ys); g = np.array(gs)
        if subset == "capped" and len(y) > cap:
            rng = np.random.default_rng(0)
            sel = _balanced_sample(y, cap, rng)
            X, y, g = X[sel], y[sel], g[sel]
        _impute_save(f"apnea_{subset}_{split}", X, y, groups=g, feat_names=F.APNEA_FEATURES)


# ---------------- T3 WESAD ----------------

def build_wesad(subset):
    win_s, ov = TASKS["wesad"]["win_s"], TASKS["wesad"]["overlap"]
    keep = TASKS["wesad"]["keep_labels"]
    cap = TASKS["wesad"]["caps"]["train_windows_per_subject"]
    Xs, ys, gs = [], [], []
    for subj in io.WESAD_SUBJECTS:
        ch, label, fs = io.load_wesad_subject(subj)
        wlen = int(win_s * fs); step = int(wlen * (1 - ov))
        sx, sy = [], []
        for s in range(0, len(label) - wlen + 1, step):
            seg_lab = label[s:s + wlen]
            u = np.unique(seg_lab)
            if len(u) == 1 and int(u[0]) in keep:        # pure window, drop transitions
                w = {k: v[s:s + wlen] for k, v in ch.items()}
                sx.append(F.wesad_window_features(w, fs)); sy.append(keep[int(u[0])])
        sx, sy = np.array(sx), np.array(sy)
        if subset == "capped" and len(sy) > cap:
            # stable per-subject seed (Python's hash() is salted per process -> not reproducible)
            seed = int(hashlib.sha256(str(subj).encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            sel = _balanced_sample(sy, cap, rng)
            sx, sy = sx[sel], sy[sel]
        Xs.append(sx); ys.append(sy); gs += [subj] * len(sy)
        print(f"  {subj}: {len(sy)} windows (pos_frac={sy.mean():.2f})")
    _impute_save(f"wesad_{subset}", np.vstack(Xs), np.concatenate(ys),
                 groups=np.array(gs), feat_names=F.WESAD_FEATURES)


# ---------------- T1 PTB-XL ----------------

def build_ptbxl(subset):
    df = io.ptbxl_metadata()
    df = df[df.n_super > 0]                               # keep labelled records
    caps = TASKS["ptbxl"]["caps"]
    classes = TASKS["ptbxl"]["classes"]
    dev = df[~df.strat_fold.isin([9, 10])]
    test = df[df.strat_fold.isin([9, 10])]
    if subset == "capped":
        rng = np.random.default_rng(0)
        dev = _ptbxl_cap(dev, classes, caps["train_per_label"] + caps["val_per_label"], rng)
        test = test.sample(n=min(caps["test_total"], len(test)), random_state=0)
    for split, sub in [("dev", dev), ("test", test)]:
        fns = sub.filename_lr.tolist()
        X = np.full((len(fns), F.PTBXL_N_FEATURES), np.nan)
        specs = []
        t = time.time()
        for i, (fn, sig) in enumerate(io.iter_ptbxl_signals(fns)):
            X[i] = F.ptbxl_features(sig)
            specs.append(F.ptbxl_spectrum(sig))
            if (i + 1) % 200 == 0:
                print(f"    {split} {i+1}/{len(fns)} ({time.time()-t:.0f}s)")
        Y = sub[classes].values.astype(np.int8)
        spec = np.vstack(specs).astype(np.float32)
        _impute_save(f"ptbxl_{subset}_{split}", X, Y,
                     folds=sub.strat_fold.values,
                     extra={"X_spectrum": spec, "ecg_id": sub.index.values},
                     feat_names=F.PTBXL_FEATURES)


def _ptbxl_cap(df, classes, per_label, rng):
    idx = set()
    for c in classes:
        pos = df.index[df[c] == 1].to_numpy()
        rng.shuffle(pos)
        idx.update(pos[:per_label].tolist())
    return df.loc[sorted(idx)]


def _balanced_sample(y, cap, rng):
    """Indices for a class-balanced subsample of size <= cap (binary)."""
    per = cap // 2
    out = []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        out.extend(ci[:per].tolist())
    out = np.array(sorted(out))
    return out


BUILDERS = {"apnea": build_apnea, "wesad": build_wesad, "ptbxl": build_ptbxl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(BUILDERS) + ["all"], required=True)
    ap.add_argument("--subset", choices=["capped", "full"], default="capped")
    ap.add_argument("--force", action="store_true", help="rebuild splits that already exist")
    a = ap.parse_args()
    global SKIP_EXISTING
    SKIP_EXISTING = not a.force
    tasks = list(BUILDERS) if a.task == "all" else [a.task]
    for t in tasks:
        print(f"== building {t} ({a.subset}) ==")
        BUILDERS[t](a.subset)


if __name__ == "__main__":
    main()
