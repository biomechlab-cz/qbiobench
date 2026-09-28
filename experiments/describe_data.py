"""Data description: sample counts per recording / subject / patient and per class, for every
pool and every hardware or screen subset.

Output: data/results/analysis_v2/data_description.json and printed tables.
Usage:  uv run python experiments/describe_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments.config import TASKS, SCREEN_CAPS, HW_PLAN      # noqa: E402
from experiments.screen import load_task_split, FEAT             # noqa: E402

OUT = ROOT / "data" / "results" / "analysis_v2"


def per_group(y, g):
    y = np.asarray(y); g = np.asarray(g).astype(str)
    df = pd.DataFrame({"g": g, "y": y if y.ndim == 1 else y.argmax(1)})
    c = df.groupby("g").size()
    pos = df.groupby("g").y.mean()
    return {"n": int(len(g)), "n_groups": int(c.size), "per_group_min": int(c.min()),
            "per_group_median": float(c.median()), "per_group_max": int(c.max()),
            "groups_all_one_class": int(((pos == 0) | (pos == 1)).sum())}


def class_counts(y, classes=None):
    y = np.asarray(y)
    if y.ndim == 1:
        return {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())}
    return {c: int(y[:, j].sum()) for j, c in enumerate(classes)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    # raw pools and missingness
    for name in ["apnea_capped_train", "apnea_capped_test", "ptbxl_capped_dev", "ptbxl_capped_test", "wesad_capped"]:
        d = np.load(FEAT / f"{name}.npz", allow_pickle=True)
        X = d["X"]
        rep[name] = {"n": int(len(X)), "n_features": int(X.shape[1]),
                     "nan_fraction": float(np.isnan(X).mean()),
                     "rows_with_nan": int(np.isnan(X).any(1).sum())}
    # WESAD per subject and class (LOSO pool)
    d = np.load(FEAT / "wesad_capped.npz", allow_pickle=True)
    ws = pd.DataFrame({"subject": d["groups"], "y": d["y"]}).groupby("subject").y.agg(["size", "sum"])
    rep["wesad_per_subject"] = {s: {"windows": int(r["size"]), "stress": int(r["sum"])} for s, r in ws.iterrows()}
    # subsets used by the screen and by hardware
    subsets = {}
    for task in ("apnea", "ptbxl", "wesad"):
        cls = TASKS[task].get("classes")
        cap = SCREEN_CAPS[task]
        for s in (0, 1, 2):
            Xtr, ytr, Xte, yte, nq, ml, gtr, gte = load_task_split(task, cap["train"], cap["test"], s, return_groups=True)
            subsets[f"screen/{task}/seed{s}"] = {"train": {**per_group(ytr, gtr), **class_counts(ytr, cls)},
                                                 "test": {**per_group(yte, gte), **class_counts(yte, cls)}}
        if task in HW_PLAN:
            for c in HW_PLAN[task]:
                ntr = c.get("n_train", c.get("n_test")); nte = c["n_test"]
                Xtr, ytr, Xte, yte, nq, ml, gtr, gte = load_task_split(task, ntr, nte, 0, return_groups=True)
                subsets[f"hardware/{task}/{c['model']}"] = {"train": {**per_group(ytr, gtr), **class_counts(ytr, cls)},
                                                            "test": {**per_group(yte, gte), **class_counts(yte, cls)}}
        Xtr, ytr, Xp, yp, nq, ml, gtr, gp = load_task_split(task, None, None, 0, return_groups=True)
        subsets[f"pool/{task}"] = {"train": {**per_group(ytr, gtr), **class_counts(ytr, cls)},
                                   "test": {**per_group(yp, gp), **class_counts(yp, cls)}}
    rep["subsets"] = subsets
    (OUT / "data_description.json").write_text(json.dumps(rep, indent=1))
    for k, v in subsets.items():
        print(f"{k:24s} train {v['train']}\n{'':24s} test  {v['test']}")
    print({k: rep[k] for k in rep if k.endswith(("train", "test", "dev", "capped"))})


if __name__ == "__main__":
    main()
