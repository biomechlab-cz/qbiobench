"""Hardware-native analysis. Every number comes from per-sample HARDWARE predictions.

Input : data/results/hw/<tag>/{cells.csv, predictions.csv, matrices/*.npz} (hw_replay2.py)
Output: data/results/analysis_v2/<tag>_pool/ (--scope pool, the full test pool) or
        data/results/analysis_v2/<tag>/ (--scope sub, the fixed test subsets, see `in_sub`)
  cells_metrics.csv   per cell: ROC-AUC with cluster and i.i.d. bootstrap CIs, PR-AUC, ECE, Brier,
                      balanced accuracy and F1 at a training-derived threshold, sizes, cost
  baselines.csv       classical models trained and scored on the EXACT inputs the QPU saw
  paired.csv          paired cluster-bootstrap differences: QML - baseline (ROC-AUC, and balanced
                      accuracy at the frozen thresholds), TREX - none, tuned - as-run, hardware -
                      closed-form classical counterpart (FQK and PQK/angle); BH-adjusted per family
  kernels.csv         spectra and centered alignment for every kernel variant
  kernel_agreement.csv hardware vs exact FQK overlaps
  pqk_feature_agreement.csv hardware vs closed-form PQK/angle projected features
  sensitivity.csv     test-size sensitivity with the trained model held fixed
  composition.csv     apnea recordings / PTB-XL patients per test set

Clusters: Apnea-ECG recording, PTB-XL patient. Thresholds: kernel cells use the threshold that
maximises balanced accuracy on out-of-fold TRAINING predictions (computed from the hardware
training kernel / features); VQC cells (trained in simulation, no hardware training predictions)
use the parity-sign threshold p = 0.5.

Usage: uv run python experiments/analysis_hw.py --tag phoenix_2026-09 --scope pool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments import stats as S                                          # noqa: E402
from experiments import kernel_diag as KD                                   # noqa: E402
from experiments.baselines import make_baselines                            # noqa: E402
from experiments.config import HW_PLAN, TASKS                               # noqa: E402
from experiments.models import ANGLE_RANGE, PQK_GRID, feature_map, _pauli   # noqa: E402
from experiments.screen import load_task_split                              # noqa: E402

B = 2000
C_GRID = {"C": [0.01, 0.1, 1, 10, 100]}


# ----------------------------------------------------------------------------- loading
SCOPE = "pool"         # "pool" = full test pool, "sub" = the fixed test subset of each cell (`in_sub`)


def cell_arrays(pred, task, model, mitig, split="test", col="score"):
    d = pred[(pred.task == task) & (pred.model == model) & (pred.mitig == mitig) & (pred.split == split)]
    if split == "test" and "in_sub" in d and SCOPE == "sub":
        d = d[d.in_sub.astype(bool)]
    labels = sorted(d.label.unique())
    Y = np.column_stack([d[d.label == l].sort_values("idx").y_true.to_numpy() for l in labels])
    P = np.column_stack([d[d.label == l].sort_values("idx")[col].to_numpy() for l in labels])
    g = d[d.label == labels[0]].sort_values("idx").group.astype(str).to_numpy()
    if labels == [-1]:
        Y, P = Y[:, 0], P[:, 0]
    return Y, P, g, labels


def _cols(Y):
    return [Y] if Y.ndim == 1 else [Y[:, j] for j in range(Y.shape[1])]


def macro(fn, Y, P):
    return float(np.mean([fn(y, p) for y, p in zip(_cols(Y), _cols(P))]))


# ----------------------------------------------------------------------------- metrics
def thresholds(Ytr, Ptr, model, n_lab):
    if model.startswith("vqc") or Ytr is None:
        return [0.5] * n_lab
    return [S.train_threshold(y, p) for y, p in zip(_cols(Ytr), _cols(Ptr))]


def metric_row(Y, Sc, P, g, th):
    """Ranking metrics and thresholds from the scores Sc; calibration from the probabilities P."""
    ci_c = S.auc_ci(Y, Sc, g, B=B); ci_i = S.auc_ci(Y, Sc, None, B=B)
    thr = [S.thresholded(y, p, t) for (y, p), t in zip(zip(_cols(Y), _cols(Sc)), th)]
    return {"roc_auc": S.macro_auc(Y, Sc), "auc_lo_cluster": ci_c["lo"], "auc_hi_cluster": ci_c["hi"],
            "auc_lo_iid": ci_i["lo"], "auc_hi_iid": ci_i["hi"],
            "roc_auc_platt": S.macro_auc(Y, P),
            "pr_auc": macro(S.pr_auc, Y, Sc), "ece": macro(S.ece, Y, P), "brier": macro(S.brier, Y, P),
            "bal_acc": float(np.mean([t["bal_acc"] for t in thr])),
            "f1": float(np.mean([t["f1"] for t in thr])), "threshold": ";".join(f"{t:.2f}" for t in th),
            "n_test": len(g), "n_clusters": len(set(g))}


# ----------------------------------------------------------------------------- baselines on the QPU inputs
def _sizes(tag, task, model):
    plan_f = ROOT / "data" / "hardware" / tag / "plan.json"
    if plan_f.exists():
        cells = json.loads(plan_f.read_text())["cells"]
        enc = model.split("-")[1] if model.startswith("pqk-") else None
        for c in cells.values():
            name = c["model"] if c["model"] != "vqc" else f"vqc-{c['vqc_train']}{'-noent' if c.get('ablate') else ''}"
            if c["model"] != "vqc" and c["encoding"] != "angle":
                name = f"{c['model']}-{c['encoding']}"
            if c["task"] == task and name == model:
                return c["n_train"], c["n_test"], np.asarray(c["sub_idx"])
    plan = next(c for c in HW_PLAN[task] if c["model"] == model)
    return plan.get("n_train", plan.get("n_test")), plan["n_test"], None


def _subsel(m, sub_idx):
    """Row selector for the pool arrays of a cell with an `in_sub` mask in the active scope."""
    if "in_sub" not in m.files or SCOPE == "pool":
        return slice(None)
    return m["in_sub"].astype(bool)


def hw_inputs(tag, task, model, mitig):
    """(Xtr_s, ytr, Xte_s) exactly as executed. Kernel cells store both sides in the PUBs; the
    VQC was trained in simulation, so its training inputs come from the (bit-exact) rebuilt
    training features and its test inputs from the PUB."""
    m = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_{model}_{mitig}.npz")
    ntr, nte, _ = _sizes(tag, task, model)
    nq = TASKS[task]["qubits"]
    Xtr, ytr, _, _, _, _ = load_task_split(task, ntr, nte, 0)
    sel = _subsel(m, None)
    if model.startswith("vqc"):
        Xtr_s = MinMaxScaler(ANGLE_RANGE).fit_transform(Xtr[:, :nq])
        return Xtr_s, ytr, m["Xte_s"], sel
    return m["Xtr_s"], ytr, m["Xte_s"], sel


CACHE = ROOT / "data" / "results" / "analysis_v2" / "_baseline_cache"


def baseline_preds_cached(Xtr, ytr, Xte, labels):
    """baseline_preds on the full executed test inputs, cached on disk by a hash of the inputs.
    Cells that share inputs (mitigation arms, VQC variants, subset and pool scopes) reuse it."""
    import hashlib
    import pickle
    h = hashlib.sha1()
    for a in (np.ascontiguousarray(Xtr, float), np.ascontiguousarray(ytr), np.ascontiguousarray(Xte, float)):
        h.update(a.tobytes()); h.update(str(a.shape).encode())
    h.update(str(labels).encode())
    f = CACHE / f"{h.hexdigest()}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = baseline_preds(Xtr, ytr, Xte, labels)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(out))
    return out


def baseline_preds(Xtr, ytr, Xte, labels):
    """Train-only CV-tuned classical models (experiments/baselines.py grids); test probabilities
    and out-of-fold training probabilities per label."""
    def score(m, X):                 # SVM: decision function (Platt can invert small-n rankings)
        return m.decision_function(X) if name == "rbf_svm" else m.predict_proba(X)[:, 1]
    out = {}
    for name, proto in make_baselines(0).items():
        Sc, P, Str = [], [], []
        for l in labels:
            y = ytr[:, l] if l >= 0 else ytr
            m = clone(proto).fit(Xtr, y)
            Sc.append(score(m, Xte)); P.append(m.predict_proba(Xte)[:, 1])
            oof = np.full(len(y), np.nan)
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y):
                oof[va] = score(clone(proto).fit(Xtr[tr], y[tr]), Xtr[va])
            Str.append(oof)
        Sc, P, Str = np.column_stack(Sc), np.column_stack(P), np.column_stack(Str)
        if labels == [-1]:
            Sc, P, Str = Sc[:, 0], P[:, 0], Str[:, 0]
        out[name] = (Sc, P, Str)
    return out


# ----------------------------------------------------------------------------- post-hoc tuned SVM
def tuned_svm(tag, task, model, mitig, ytr, labels):
    m = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_{model}_{mitig}.npz")
    P = []
    for l in labels:
        y = ytr[:, l] if l >= 0 else ytr
        sel = _subsel(m, None)
        if model == "fqk":
            g = GridSearchCV(SVC(kernel="precomputed", probability=True, random_state=0), C_GRID, cv=3)
            g.fit(m["G_tr"], y); P.append(g.best_estimator_.decision_function(m["G_te"][sel]))
        else:
            s = StandardScaler().fit(m["F_tr"])
            g = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=0), PQK_GRID, cv=3)
            g.fit(s.transform(m["F_tr"]), y); P.append(g.best_estimator_.decision_function(s.transform(m["F_te"][sel])))
    P = np.column_stack(P)
    return P[:, 0] if labels == [-1] else P


# ----------------------------------------------------------------------------- closed-form FQK
def closed_form_fqk(tag, task, mitig, ytr, labels):
    """The classical counterpart of FQK/angle, k = prod cos^2((x-y)/2), on the inputs the QPU
    saw: (a) Nystrom with the same 16 landmarks (what the hardware approximated), (b) the exact
    kernel on all training samples. Same SVC (C=1) as the as-run hardware tail."""
    m = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_fqk_{mitig}.npz")
    sel = _subsel(m, None)
    Xtr, Xte, L = m["Xtr_s"], m["Xte_s"][sel], m["landmarks"]
    Gn_tr, Gn_te = KD.nystrom(KD.fqk_angle_closed_form(Xtr, L), KD.fqk_angle_closed_form(L, L),
                              KD.fqk_angle_closed_form(Xte, L))
    Ge_tr, Ge_te = KD.fqk_angle_closed_form(Xtr, Xtr), KD.fqk_angle_closed_form(Xte, Xtr)
    res = {}
    for key, (Gtr, Gte) in {"nystrom": (Gn_tr, Gn_te), "exact": (Ge_tr, Ge_te)}.items():
        P = []
        for l in labels:
            y = ytr[:, l] if l >= 0 else ytr
            P.append(SVC(kernel="precomputed", probability=True, random_state=0).fit(Gtr, y).decision_function(Gte))
        P = np.column_stack(P)
        res[key] = P[:, 0] if labels == [-1] else P
    agree = {"task": task, "mitig": mitig}
    for blk, Kh, Kc in [("LL", m["K_LL"], KD.fqk_angle_closed_form(L, L)),
                        ("trL", m["K_trL"], KD.fqk_angle_closed_form(Xtr, L)),
                        ("teL", m["K_teL"][sel], KD.fqk_angle_closed_form(Xte, L))]:
        agree[f"{blk}_mae"] = float(np.abs(Kh - Kc).mean())
        agree[f"{blk}_corr"] = float(np.corrcoef(Kh.ravel(), Kc.ravel())[0, 1])
        agree[f"{blk}_slope"] = float(np.polyfit(Kc.ravel(), Kh.ravel(), 1)[0])
    return res, agree


# ----------------------------------------------------------------------------- closed-form PQK
def closed_form_pqk(tag, task, mitig, ytr, labels):
    """The classical counterpart of PQK/angle: the closed-form projected features
    (kernel_diag.pqk_angle_closed_form, equal to the noiseless statevector features) on the
    inputs the QPU saw, standardised on the training units and fed to the same fixed-default
    RBF-SVM as the as-run hardware tail. Also returns the agreement of the measured <X_i> and
    <Z_i> columns with their closed form (the <Y_i> columns are zero in closed form)."""
    m = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_pqk_{mitig}.npz")
    sel = _subsel(m, None)
    Ftr, Fte = KD.pqk_angle_closed_form(m["Xtr_s"]), KD.pqk_angle_closed_form(m["Xte_s"][sel])
    s = StandardScaler().fit(Ftr)
    P = []
    for l in labels:
        y = ytr[:, l] if l >= 0 else ytr
        P.append(SVC(kernel="rbf", probability=True, random_state=0).fit(s.transform(Ftr), y)
                 .decision_function(s.transform(Fte)))
    P = np.column_stack(P)
    agree = {"task": task, "mitig": mitig}
    for side, Fh, Fc in (("tr", m["F_tr"], Ftr), ("te", m["F_te"][sel], Fte)):
        for name, cols in (("X", slice(0, None, 3)), ("Z", slice(2, None, 3))):
            agree[f"{side}{name}_corr"] = float(np.corrcoef(Fh[:, cols].ravel(), Fc[:, cols].ravel())[0, 1])
            agree[f"{side}{name}_mae"] = float(np.abs(Fh[:, cols] - Fc[:, cols]).mean())
        agree[f"{side}Y_sd"] = float(Fh[:, 1::3].std())
    return (P[:, 0] if labels == [-1] else P), agree


def binarize(Sc, th):
    """Frozen binary predictions at the training-derived thresholds, one per label."""
    H = [(c >= t).astype(int) for c, t in zip(_cols(np.asarray(Sc)), th)]
    return H[0] if np.asarray(Sc).ndim == 1 else np.column_stack(H)


# ----------------------------------------------------------------------------- kernel diagnostics
def _sv_projected(X, nq):
    from qiskit.quantum_info import Statevector
    fm = feature_map("angle", nq)
    obs = [_pauli(Lt, i, nq) for i in range(nq) for Lt in "XYZ"]
    F = np.empty((len(X), 3 * nq))
    for k, x in enumerate(X):
        st = Statevector(fm.assign_parameters(x))
        F[k] = [st.expectation_value(o).real for o in obs]
    return F


def kernel_rows(tag, task):
    rows = []
    mf = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_fqk_none.npz")
    mp = np.load(ROOT / "data" / "results" / "hw" / tag / "matrices" / f"{task}_pqk_none.npz")
    nq = TASKS[task]["qubits"]
    _, yf, _, _, _, _ = load_task_split(task, 64, 80, 0)
    _, yp, _, _, _, _ = load_task_split(task, 256, 416, 0)
    Xf, L = mf["Xtr_s"], mf["landmarks"]
    K_exact = KD.fqk_angle_closed_form(Xf, Xf)
    K_nys, _ = KD.nystrom(KD.fqk_angle_closed_form(Xf, L), KD.fqk_angle_closed_form(L, L))
    F_sv = _sv_projected(mp["Xtr_s"], nq)
    Fs_sv = StandardScaler().fit_transform(F_sv)
    Fs_hw = StandardScaler().fit_transform(mp["F_tr"])
    K_rbf_sv, g_sv = KD.rbf_gram(Fs_sv); K_rbf_hw, g_hw = KD.rbf_gram(Fs_hw)
    zero_y = float(np.abs(F_sv[:, 1::3]).max())                     # <Y_i> columns
    variants = [
        ("FQK/angle", "exact (statevector = closed form)", K_exact, yf, "noiseless"),
        ("FQK/angle", "Nystrom L=16 (noiseless)", K_nys, yf, "noiseless"),
        ("FQK/angle", "Nystrom L=16 (hardware)", mf["G_tr"], yf, "hardware"),
        ("PQK/angle", f"RBF on standardised features (noiseless, gamma={g_sv:.3g})", K_rbf_sv, yp, "noiseless"),
        ("PQK/angle", f"RBF on standardised features (hardware, gamma={g_hw:.3g})", K_rbf_hw, yp, "hardware"),
        ("PQK/angle", "linear Gram of raw features", F_sv @ F_sv.T, yp, "noiseless"),
    ]
    for model, kind, K, y, src in variants:
        st = KD.spectrum_stats(K)
        rows.append({"task": task, "model": model, "kernel": kind, "source": src,
                     "kta_centered": KD.macro_alignment(K, y), **st,
                     "pqk_max_abs_Y": zero_y if model == "PQK/angle" else None})
    return rows


# ----------------------------------------------------------------------------- sensitivity
def sensitivity(Y, P, g, sizes, reps=1000, seed=0):
    rng = np.random.default_rng(seed); out = []
    n = len(g)
    for s in sizes:
        if s > n:
            continue
        vals = []
        for _ in range(reps if s < n else 1):
            idx = rng.choice(n, s, replace=False)
            try:
                vals.append(S.macro_auc(Y[idx], P[idx]))
            except ValueError:
                continue
        v = np.asarray(vals)
        out.append({"n": s, "mean": float(v.mean()), "sd": float(v.std()), "p2.5": float(np.percentile(v, 2.5)),
                    "p97.5": float(np.percentile(v, 97.5)), "reps": len(v)})
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phoenix_2026-09")
    ap.add_argument("--scope", choices=["sub", "pool"], default="pool",
                    help="full test pool, or the fixed test subset of each cell (predictions.csv `in_sub`: 80 units "
                         "for FQK, 416 for PQK, 500 PTB-XL records for the VQC)")
    a = ap.parse_args()
    global SCOPE
    SCOPE = a.scope
    src = ROOT / "data" / "results" / "hw" / a.tag
    out = ROOT / "data" / "results" / "analysis_v2" / (a.tag if a.scope == "sub" else f"{a.tag}_pool")
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(src / "predictions.csv"); cells = pd.read_csv(src / "cells.csv")
    mrows, brows, prows, agree_rows, sens_rows, comp_rows, pagree_rows = [], [], [], [], [], [], []
    for _, c in cells.iterrows():
        task, model, mitig = c.task, c.model, c.mitig
        Y, P, g, labels = cell_arrays(pred, task, model, mitig, col="score")      # ranking scores
        _, Pp, _, _ = cell_arrays(pred, task, model, mitig, col="proba")           # probabilities
        has_tr = not model.startswith("vqc")
        Ytr = Ptr = None
        if has_tr:
            Ytr, Ptr, _, _ = cell_arrays(pred, task, model, mitig, "train_oof", col="score")
        th = thresholds(Ytr, Ptr, model, len(labels))
        mrows.append({"task": task, "model": model, "encoding": c.encoding, "mitig": mitig,
                      "source": "hardware", "backend": c.backend, **metric_row(Y, P, Pp, g, th),
                      "n_train": c.n_train, "circuits": c.circuits, "qpu_min": c.qpu_min, "bell": c.bell})
        print(f"{task} {model:3s} {mitig:4s} AUC {mrows[-1]['roc_auc']:.3f} "
              f"[{mrows[-1]['auc_lo_cluster']:.3f},{mrows[-1]['auc_hi_cluster']:.3f}] "
              f"ECE {mrows[-1]['ece']:.3f} balacc {mrows[-1]['bal_acc']:.3f}", flush=True)
        # classical baselines on the identical executed inputs
        Xtr_s, ytr, Xte_full, sel = hw_inputs(a.tag, task, model, mitig)
        base = baseline_preds_cached(Xtr_s, ytr, Xte_full, labels)
        Hq = binarize(P, th)
        for name, (Pb, Pbp, Pbtr) in base.items():
            Pb, Pbp = Pb[sel], Pbp[sel]
            thb = [S.train_threshold(y, p) for y, p in zip(_cols(ytr[:, labels] if labels != [-1] else ytr), _cols(Pbtr))]
            brows.append({"task": task, "cell": f"{model}/{mitig}", "baseline": name, **metric_row(Y, Pb, Pbp, g, thb)})
            d = S.paired_auc_diff(Y, P, Pb, g, B=B)
            di = S.paired_auc_diff(Y, P, Pb, None, B=B)
            prows.append({"family": "qml_vs_baseline" + ("" if mitig == "none" else "_trex"),
                          "task": task, "a": f"{model}/{mitig}", "b": name, **d,
                          "lo_iid": di["lo"], "hi_iid": di["hi"], "primary": name == "rbf_svm"})
            # balanced accuracy at the training-derived thresholds, paired on the same test samples
            d = S.paired_ba_diff(Y, Hq, binarize(Pb, thb), g, B=B)
            prows.append({"family": "qml_vs_baseline_ba" + ("" if mitig == "none" else "_trex"),
                          "task": task, "a": f"{model}/{mitig}", "b": name, **d, "primary": name == "rbf_svm"})
        # mitigation effect (paired, same samples)
        if mitig == "trex":
            Y0, P0, g0, _ = cell_arrays(pred, task, model, "none", col="score")
            assert np.array_equal(Y0, Y)
            d = S.paired_auc_diff(Y, P, P0, g, B=B)
            q0 = cells[(cells.task == task) & (cells.model == model) & (cells.mitig == "none")].qpu_min.iloc[0]
            prows.append({"family": "trex_minus_none", "task": task, "a": f"{model}/trex", "b": f"{model}/none",
                          **d, "extra_qpu_min": float(c.qpu_min - q0), "primary": True})
        # post-hoc matched hyperparameter search for the kernel SVMs
        if model in ("fqk", "pqk", "pqk-zz"):
            Pt = tuned_svm(a.tag, task, model, mitig, ytr, labels)
            d = S.paired_auc_diff(Y, Pt, P, g, B=B)
            prows.append({"family": "tuned_minus_asrun", "task": task, "a": f"{model}/{mitig} tuned",
                          "b": f"{model}/{mitig} as-run", "auc_tuned": S.macro_auc(Y, Pt), **d, "primary": False})
        if model == "fqk":
            cf, agree = closed_form_fqk(a.tag, task, mitig, ytr, labels)
            agree_rows.append(agree)
            for key, Pc in cf.items():
                d = S.paired_auc_diff(Y, P, Pc, g, B=B)
                prows.append({"family": "hardware_minus_classical_fqk", "task": task, "a": f"fqk/{mitig} hardware",
                              "b": f"closed-form {key}", "auc_b": S.macro_auc(Y, Pc), **d, "primary": key == "nystrom"})
        if model == "pqk":
            Pc, agree = closed_form_pqk(a.tag, task, mitig, ytr, labels)
            pagree_rows.append(agree)
            d = S.paired_auc_diff(Y, P, Pc, g, B=B)
            prows.append({"family": "hardware_minus_classical_pqk", "task": task, "a": f"pqk/{mitig} hardware",
                          "b": "closed-form features", "auc_b": S.macro_auc(Y, Pc), **d, "primary": True})
        if mitig == "none":
            n_all = len(g)
            sizes = sorted({s_ for s_ in [20, 40, 80, 160, 240, 320, 416, 500, n_all] if s_ <= n_all})
            for r in sensitivity(Y, P, g, sizes):
                sens_rows.append({"task": task, "model": model, "source": "hardware", **r})
            comp_rows.append({"task": task, "cell": model, "n_test": len(g), "n_clusters": len(set(g)),
                              "per_cluster_min": int(pd.Series(g).value_counts().min()),
                              "per_cluster_median": float(pd.Series(g).value_counts().median()),
                              "per_cluster_max": int(pd.Series(g).value_counts().max()),
                              "pos_frac": float(np.mean(Y if Y.ndim == 1 else Y.mean(1)))})
    # multiplicity: BH within each family (primary comparisons only for the QML-vs-baseline family)
    pdf = pd.DataFrame(prows)
    pdf["p_bh"] = np.nan
    for fam, sub in pdf.groupby("family"):
        m = sub.primary.astype(bool)
        if m.any():
            adj, _ = S.bh(sub.loc[m, "p_boot"].to_numpy())
            pdf.loc[sub.index[m], "p_bh"] = adj
    pd.DataFrame(mrows).to_csv(out / "cells_metrics.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "baselines.csv", index=False)
    pdf.to_csv(out / "paired.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(out / "kernel_agreement.csv", index=False)
    pd.DataFrame(pagree_rows).to_csv(out / "pqk_feature_agreement.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(out / "composition.csv", index=False)
    krows = []
    for task in ("apnea", "ptbxl"):
        krows += kernel_rows(a.tag, task)
    pd.DataFrame(krows).to_csv(out / "kernels.csv", index=False)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
