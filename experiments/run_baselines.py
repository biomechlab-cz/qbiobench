"""Phase 1 (executing) — same-feature classical baselines (the FAIR comparison).

Tier-1 estimators (logreg/RBF-SVM/XGBoost/MLP) on the SAME train-only-PCA features the
QML models will use. Each task uses its split protocol:
  apnea  — per-recording train/test (a,b,c → x); bootstrap CI on test predictions
  wesad  — LOSO; subject-aggregated metrics, bootstrap over subjects
  ptbxl  — one-vs-rest multilabel; PCA fit on dev, macro-AUC/PR-AUC on folds 9–10 test

Writes data/results/baselines_<task>.csv + a run manifest. No IBM cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.preprocessing import FeatureReducer        # noqa: E402
from utils.splits import loso                          # noqa: E402
from experiments.baselines import make_baselines       # noqa: E402
from experiments import metrics as M                    # noqa: E402
from experiments.config import TASKS                    # noqa: E402
import qbio_runtime as rt                                # noqa: E402

FEAT = ROOT / "data" / "features"
RES = ROOT / "data" / "results"


def _load(name):
    d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
    return d


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.ptp() + 1e-12)


def run_apnea(subset):
    tr, te = _load(f"apnea_{subset}_train"), _load(f"apnea_{subset}_test")
    Xtr, ytr = tr["X"], tr["y"]; Xte, yte = te["X"], te["y"]
    fr = FeatureReducer(TASKS["apnea"]["pca_dim"]).fit(Xtr)
    Xtr_p, Xte_p = fr.transform(Xtr), fr.transform(Xte)
    rows = []
    for name, model in make_baselines(0).items():
        model.fit(Xtr_p, ytr)
        p = _proba(model, Xte_p)
        m = M.task_metrics(yte, p)
        mean, lo, hi = M.bootstrap_ci(yte, p, lambda a, b: M.task_metrics(a, b)["roc_auc"])
        m.update({"ece": M.expected_calibration_error(yte, p), "brier": M.brier(yte, p),
                  "roc_auc_lo": lo, "roc_auc_hi": hi, "model": name})
        rows.append(m)
    return pd.DataFrame(rows)


def run_wesad(subset):
    d = _load(f"wesad_{subset}")
    X, y, g = d["X"], d["y"], d["groups"]
    per_subject = {name: [] for name in make_baselines(0)}
    for tr_idx, te_idx in loso(g):
        fr = FeatureReducer(TASKS["wesad"]["pca_dim"]).fit(X[tr_idx])
        Xtr, Xte = fr.transform(X[tr_idx]), fr.transform(X[te_idx])
        for name, model in make_baselines(0).items():
            model.fit(Xtr, y[tr_idx])
            p = _proba(model, Xte)
            per_subject[name].append(M.task_metrics(y[te_idx], p)["balanced_acc"])
    rows = []
    for name, accs in per_subject.items():
        accs = np.array(accs)
        rows.append({"model": name, "balanced_acc": accs.mean(),
                     "balanced_acc_std": accs.std(), "n_subjects": len(accs),
                     "balanced_acc_lo": np.percentile(accs, 2.5),
                     "balanced_acc_hi": np.percentile(accs, 97.5)})
    return pd.DataFrame(rows)


def run_ptbxl(subset):
    dev, te = _load(f"ptbxl_{subset}_dev"), _load(f"ptbxl_{subset}_test")
    Xdev, Ydev = dev["X"], dev["y"]; Xte, Yte = te["X"], te["y"]
    classes = TASKS["ptbxl"]["classes"]
    fr = FeatureReducer(TASKS["ptbxl"]["pca_dim"]).fit(Xdev)
    Xdev_p, Xte_p = fr.transform(Xdev), fr.transform(Xte)
    rows = []
    for name, _ in make_baselines(0).items():
        probas = np.zeros_like(Yte, dtype=float)
        for c in range(len(classes)):
            model = make_baselines(0)[name]
            if len(np.unique(Ydev[:, c])) < 2:
                continue
            model.fit(Xdev_p, Ydev[:, c])
            probas[:, c] = _proba(model, Xte_p)
        m = M.task_metrics(Yte, probas, multilabel=True)
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows)


RUNNERS = {"apnea": run_apnea, "wesad": run_wesad, "ptbxl": run_ptbxl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(RUNNERS) + ["all"], required=True)
    ap.add_argument("--subset", choices=["capped", "full"], default="capped")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    for t in (list(RUNNERS) if a.task == "all" else [a.task]):
        print(f"== baselines: {t} ({a.subset}) ==")
        df = RUNNERS[t](a.subset).round(4)
        print(df.to_string(index=False))
        df.to_csv(RES / f"baselines_{t}_{a.subset}.csv", index=False)
        rt.write_manifest(f"baselines_{t}_{a.subset}",
                          {"task": t, "subset": a.subset, "tier": "same-feature"},
                          {"results": df.to_dict(orient="records")}, subdir="manifests")


if __name__ == "__main__":
    main()
