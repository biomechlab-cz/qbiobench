"""Phase 2 - simulation screen.

Runs the 7 method cells x 3 seeds per task, ranks them, and applies the promotion rule.

Modes:
  sv     - noiseless statevector on the logical circuits (reference ranking)
  noisy  - device noise: every circuit is transpiled onto the target of the noise snapshot of an
           IBM Heron r3 processor (June 2026, data/noise/heron_r3_2026-06-11.pkl; error-aware
           layout, device basis) and simulated with the matching noise model
           (experiments/quantum_backend.NoisyCircuit). No error mitigation.

Protocol:
  * caps: apnea/WESAD train 128 / test 128, PTB-XL train 64 / test 128
  * PQK: RBF SVM on standardised projected features, (gamma, C) by 3-fold CV on train
  * FQK: exact kernel in sv; Nystrom with 32 landmarks from 4096-shot overlap circuits in noisy
  * VQC: SPSA (200 iterations) on binary cross-entropy of p = (1 + <Z..Z>)/2, parameters
    trained noiselessly and evaluated under the target noise;
    `--vqc-train noisy` is the noise-aware control, `--vqc-loss one-sided` the one-sided
    objective (see models.py), screened for the comparison of the two objectives
  * entanglement-removed control for every VQC and FQK cell. For FQK/angle it is identical by
    construction (the CZ ring cancels in U(x)U(y)^dagger), which is checked, not assumed.
  * kernels are computed once per (cell, seed) and reused across the PTB-XL one-vs-rest
    labels; a seeded kernel is label independent, so this matches per-label runs.

Promotion rule: for each primary task, the encoding with the highest mean
ROC-AUC within each model family (PQK, FQK, VQC) is promoted.

Test data: the screen draws its test units from the same capped test pools that the hardware
evaluation scores (apnea recordings x01-x35, PTB-XL folds 9-10), so the screen ranking and the
promotion saw test data.

Usage:
  uv run python experiments/screen.py --task all --mode noisy --jobs 28 --tag screen_v2
  uv run python experiments/screen.py --task all --mode sv    --jobs 28 --tag screen_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from utils.preprocessing import FeatureReducer            # noqa: E402
from experiments.config import TASKS, METHOD_CELLS, SEEDS, SCREEN_CAPS  # noqa: E402

FEAT = ROOT / "data" / "features"
RES = ROOT / "data" / "results"


# ----------------------------------------------------------------------------- data
def _ptbxl_patients(ecg_ids):
    from utils import io
    df = io.ptbxl_metadata()
    return df.loc[np.asarray(ecg_ids, dtype=int), "patient_id"].to_numpy().astype(int)


def _raw_pools(task, want_groups):
    if task == "apnea":
        tr = np.load(FEAT / "apnea_capped_train.npz"); te = np.load(FEAT / "apnea_capped_test.npz")
        return tr["X"], tr["y"], te["X"], te["y"], tr["groups"], te["groups"]
    if task == "ptbxl":
        tr = np.load(FEAT / "ptbxl_capped_dev.npz"); te = np.load(FEAT / "ptbxl_capped_test.npz")
        if want_groups:
            gtr, gte = _ptbxl_patients(tr["ecg_id"]), _ptbxl_patients(te["ecg_id"])
        else:
            gtr, gte = np.zeros(len(tr["y"])), np.zeros(len(te["y"]))
        return tr["X"], tr["y"], te["X"], te["y"], gtr, gte
    d = np.load(FEAT / "wesad_capped.npz", allow_pickle=True)
    g = d["groups"]; subs = sorted(set(g.tolist()))
    te_subs = set(subs[-5:])                                # grouped holdout for screening
    m = np.array([s in te_subs for s in g])
    return d["X"][~m], d["y"][~m], d["X"][m], d["y"][m], g[~m], g[m]


def load_task_split(task, train_cap=None, test_cap=None, seed=0, *, return_groups=False,
                    return_val=False, val_cap=128):
    """Return (Xtr, ytr, Xte, yte, n_qubits, multilabel) with train-only PCA applied.
    train_cap/test_cap subsample (None = use all). The train subsample is drawn first and the
    test subsample second, from one seeded generator, so both depend on (train_cap, seed).

    return_groups=True appends the cluster id of every train and test sample: the Apnea-ECG
    recording, the PTB-XL patient, or the WESAD subject (units of the cluster bootstrap).
    return_val=True appends (Xval, yval): up to val_cap samples of the training pool that were
    NOT drawn into the training subsample (disjoint from train and test), used only for VQC
    learning curves. It does not change the train/test draws."""
    spec = TASKS[task]; nq = spec["qubits"]; pdim = spec["pca_dim"]
    ml = spec["type"] == "multilabel"
    Xtr_r, ytr, Xte_r, yte, gtr, gte = _raw_pools(task, return_groups)
    fr = FeatureReducer(pdim, seed=seed).fit(Xtr_r)
    Xtr, Xte = fr.transform(Xtr_r), fr.transform(Xte_r)
    rng = np.random.default_rng(seed)
    unused = np.arange(len(ytr))
    if train_cap and len(ytr) > train_cap:
        i = rng.choice(len(ytr), train_cap, replace=False)
        unused = np.setdiff1d(np.arange(len(ytr)), i)
        Xval_pool, yval_pool = Xtr[unused], ytr[unused]
        Xtr, ytr, gtr = Xtr[i], ytr[i], gtr[i]
    else:
        Xval_pool, yval_pool = Xtr[:0], ytr[:0]
    if test_cap and len(yte) > test_cap:
        i = rng.choice(len(yte), test_cap, replace=False); Xte, yte, gte = Xte[i], yte[i], gte[i]
    out = [Xtr, ytr, Xte, yte, nq, ml]
    if return_groups:
        out += [np.asarray(gtr), np.asarray(gte)]
    if return_val:
        vr = np.random.default_rng(seed + 1000)
        k = min(val_cap, len(yval_pool))
        j = vr.choice(len(yval_pool), k, replace=False) if k else np.arange(0)
        out += [Xval_pool[j], yval_pool[j]]
    return tuple(out)


# ----------------------------------------------------------------------------- one unit
def _labels(y, ml):
    if not ml:
        return [(None, y)]
    return [(c, y[:, c]) for c in range(y.shape[1]) if len(np.unique(y[:, c])) > 1]


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def run_unit(task, model, encoding, seed, mode, vqc_train="sv", hist_dir=None, vqc_loss="bce"):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from experiments.models import PQKQSVM, FQKQSVM, VQCReuploading, PQK_GRID
    cap = SCREEN_CAPS[task]
    Xtr, ytr, Xte, yte, nq, ml, Xv, yv = load_task_split(task, cap["train"], cap["test"], seed,
                                                         return_val=True)
    t0 = time.time()
    out = {"task": task, "model": model, "encoding": encoding, "seed": seed, "mode": mode,
           "vqc_train": (vqc_train if model == "vqc" else None),
           "vqc_loss": (vqc_loss if model == "vqc" else None)}

    def kernel_cell(ablate):
        if model == "pqk":
            m = PQKQSVM(encoding, nq, mode=mode, seed=seed, ablate_ent=ablate)
            from sklearn.preprocessing import MinMaxScaler
            from experiments.models import ANGLE_RANGE
            m.in_scaler = MinMaxScaler(ANGLE_RANGE).fit(np.asarray(Xtr, float)[:, :nq])
            Ftr, Fte = m.projected_features(Xtr), m.projected_features(Xte)
            s = StandardScaler().fit(Ftr); Ftr, Fte = s.transform(Ftr), s.transform(Fte)
            aucs = []
            for c, yc in _labels(ytr, ml):
                g = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=seed), PQK_GRID, cv=3, n_jobs=1)
                p = g.fit(Ftr, yc).best_estimator_.predict_proba(Fte)[:, 1]
                aucs.append(_auc(yte[:, c] if c is not None else yte, p))
            return aucs, (m._nc if mode == "noisy" else None)
        m = FQKQSVM(encoding, nq, mode=mode, seed=seed, ablate_ent=ablate)
        y0 = _labels(ytr, ml)[0][1]
        m.fit(Xtr, y0)                                   # computes and stores the train Gram
        Xte_s = m.in_scaler.transform(np.asarray(Xte, float)[:, :nq])
        Gte = (m.kernel(Xte_s, m.L) @ m.Wh) @ m.Phi_tr.T if m.nystrom else m.kernel(Xte_s, m.Xtr)
        aucs = []
        for c, yc in _labels(ytr, ml):
            svm = SVC(kernel="precomputed", probability=True, random_state=seed).fit(m.G_tr, yc)
            aucs.append(_auc(yte[:, c] if c is not None else yte, svm.predict_proba(Gte)[:, 1]))
        return aucs, m._nc

    def vqc_cell(ablate):
        aucs, nc = [], None
        yv_list = dict(_labels(yv, ml)) if len(yv) else {}
        for c, yc in _labels(ytr, ml):
            m = VQCReuploading(encoding, nq, mode=mode, train_mode=vqc_train, seed=seed,
                               ablate_ent=ablate, loss=vqc_loss)
            yvc = yv_list.get(c) if ml else (yv if len(yv) else None)
            m.fit(Xtr, yc, X_val=(Xv if yvc is not None else None), y_val=yvc)
            aucs.append(_auc(yte[:, c] if c is not None else yte, m.predict_proba(Xte)))
            nc = nc or m._nc
            if hist_dir is not None:
                hist_dir.mkdir(parents=True, exist_ok=True)
                tagc = (f"{task}_{model}_{encoding}_s{seed}_{mode}_tr{vqc_train}_{vqc_loss}"
                        f"{'_noent' if ablate else ''}_c{c if c is not None else 0}")
                (hist_dir / f"{tagc}.json").write_text(json.dumps(
                    {"history": m.history, "fit": m.fit_result, "n_weights": m.n_weights,
                     "test_auc": aucs[-1]}))
        return aucs, nc

    fn = vqc_cell if model == "vqc" else kernel_cell
    aucs, nc = fn(False)
    out["roc_auc"] = float(np.mean(aucs))
    out["per_label_auc"] = ";".join(f"{a:.4f}" for a in aucs)
    do_abl = model == "vqc" or model == "fqk"
    if do_abl:
        a2, _ = fn(True)
        out["roc_auc_noent"] = float(np.mean(a2))
        out["ent_sensitivity"] = out["roc_auc"] - out["roc_auc_noent"]
    if nc is not None:
        out["depth"], out["n_2q"], out["phys_qubits"] = nc.depth, nc.two_qubit_gates, ",".join(map(str, nc.phys))
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------- ranking
def rank(df):
    agg = (df.groupby(["task", "model", "encoding"])
             .agg(auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
                  auc_noent_mean=("roc_auc_noent", "mean"),
                  n_2q=("n_2q", "first") if "n_2q" in df else ("roc_auc", "size"),
                  runtime_s=("runtime_s", "mean"))
             .reset_index().sort_values(["task", "auc_mean"], ascending=[True, False]))
    return agg


def promote(agg, tasks=("apnea", "ptbxl")):
    """Best mean-AUC encoding within each model family, per primary task."""
    rows = []
    for t in tasks:
        a = agg[agg.task == t]
        for mdl in ("fqk", "pqk", "vqc"):
            b = a[a.model == mdl].sort_values("auc_mean", ascending=False)
            if len(b):
                r = b.iloc[0]
                runner = b.iloc[1] if len(b) > 1 else None
                rows.append({"task": t, "model": mdl, "encoding": r.encoding, "auc_mean": r.auc_mean,
                             "margin_to_next": (r.auc_mean - runner.auc_mean) if runner is not None else None})
    return rows


def main():
    ap = argparse.ArgumentParser(description="QBioBench simulation screen")
    ap.add_argument("--task", nargs="+", choices=list(TASKS) + ["all"], required=True)
    ap.add_argument("--mode", choices=["sv", "noisy"], default="noisy")
    ap.add_argument("--vqc-train", choices=["sv", "noisy"], default="sv")
    ap.add_argument("--vqc-loss", choices=["bce", "one-sided", "legacy"], default="bce",
                    help="bce = binary cross-entropy; one-sided = the one-sided cross-entropy objective "
                         "(models.py), for the comparison of the two objectives; 'legacy' is an alias of "
                         "one-sided and the name used in its output files")
    ap.add_argument("--models", nargs="*", default=["pqk", "fqk", "vqc"])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default="screen_v2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true", help="re-rank the logged units, run nothing")
    a = ap.parse_args()
    if a.vqc_loss == "one-sided":                         # output files and unit logs keep the name "legacy"
        a.vqc_loss = "legacy"
    tasks = ["apnea", "wesad", "ptbxl"] if "all" in a.task else a.task
    units = [(t, m, e, s) for t in tasks for (m, e) in METHOD_CELLS if m in a.models for s in a.seeds]
    print(f"{len(units)} units ({a.mode}, vqc_train={a.vqc_train}, jobs={a.jobs})", flush=True)
    if a.dry_run:
        for u in units:
            print("  ", u)
        return
    import pandas as pd
    from joblib import Parallel, delayed
    out_dir = RES / "screen" / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    hist = out_dir / "vqc_histories"
    os.environ.setdefault("QBIO_AER_THREADS", "1" if a.jobs > 1 else "0")
    name = f"{a.mode}_vqc{a.vqc_train}" + ("" if a.vqc_loss == "bce" else f"_{a.vqc_loss}")
    parts = {t: out_dir / f"{name}_{t}_units.jsonl" for t in tasks}   # one crash-safe log per task
    def _parts():                                     # exact names: "sv_vqcsv" must not match "sv_vqcsv_legacy"
        return [out_dir / f"{name}_{t}_units.jsonl" for t in TASKS if (out_dir / f"{name}_{t}_units.jsonl").exists()]
    done = set()
    for p in _parts():
        for line in p.read_text().splitlines():
            r = json.loads(line); done.add((r["task"], r["model"], r["encoding"], r["seed"]))
    todo = [u for u in units if u not in done]
    print(f"  {len(done)} already done, {len(todo)} to run", flush=True)

    def work(u):
        r = run_unit(*u, a.mode, a.vqc_train, hist, a.vqc_loss)
        with open(parts[u[0]], "a") as f:                # crash-safe incremental log
            f.write(json.dumps(r) + "\n")
        noent = f" (no-ent {r['roc_auc_noent']:.4f})" if "roc_auc_noent" in r else ""
        print(f"  {r['task']:5s} {r['model']}/{r['encoding']:11s} s{r['seed']}: AUC {r['roc_auc']:.4f}"
              f"{noent}  2q={r.get('n_2q')}  {r['runtime_s']}s", flush=True)
        return r

    if not a.aggregate_only:
        Parallel(n_jobs=a.jobs, backend="loky", verbose=0)(delayed(work)(u) for u in todo)
    rows = [json.loads(l) for p in _parts() for l in p.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{name}_raw.csv", index=False)
    agg = rank(df)
    agg.to_csv(out_dir / f"{name}_ranked.csv", index=False)
    print("\n=== RANKED ===")
    print(agg.to_string(index=False))
    pr = promote(agg)
    (out_dir / f"{name}_promotion.json").write_text(json.dumps(pr, indent=1, default=float))
    print("\n=== PROMOTION (best encoding per model family) ===")
    for r in pr:
        print(f"  {r['task']:5s} {r['model']}: {r['encoding']} ({r['auc_mean']:.3f}; margin {r['margin_to_next']})")


if __name__ == "__main__":
    main()
