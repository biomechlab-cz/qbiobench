"""Plan-driven replay of hw_run2.py hardware runs (IBM Phoenix): recompute every hardware result from
the archived raw job records (no QPU time), on the full test pool.

Input : data/hardware/<tag>/plan.json   (cell -> role -> job id, subset indices)
        data/hardware/<tag>/<job>.json  (archived by hw_archive.py --jobs ...; .json.gz accepted)
Output: data/results/hw/<tag>/{cells.csv, predictions.csv, matrices/*.npz}
        predictions.csv carries `in_sub` = the sample belongs to the fixed test subset of the cell
        (plan.json `sub_idx`: the seeded n_test draw of load_task_split, 80 units for FQK, 416 for
        PQK, 500 PTB-XL records for the VQC, the whole 510-minute pool for the Apnea-ECG VQC), so
        the subset metric and the full-pool metric come from the same measurements and the same
        trained model.

Usage: uv run python experiments/hw_replay2.py --tag phoenix_2026-09
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments.config import TASKS                                    # noqa: E402
from experiments.models import ANGLE_RANGE                              # noqa: E402
from experiments.hw_archive import load_archived, HW_DIR                # noqa: E402
from experiments.hw_run2 import cell_data                               # noqa: E402

OUT_ROOT = ROOT / "data" / "results" / "hw"


# ----------------------------------------------------------------------------- measurements
def _counts_p0(pubres, nq):
    creg = pubres.data.c if hasattr(pubres.data, "c") else pubres.data.meas
    zero = "0" * nq
    out = np.empty(creg.size if hasattr(creg, "size") else len(creg))
    for j in range(len(out)):
        cnt = creg[j].get_counts()
        out[j] = cnt.get(zero, 0) / max(1, sum(cnt.values()))
    return out


# ----------------------------------------------------------------------------- classical tail
def _oof_proba(G_or_F, y, kernel, seed=0, **svc_kw):
    """Out-of-fold training probabilities and decision scores (5-fold stratified), used only for
    threshold selection on training data."""
    oof, oofs = np.full(len(y), np.nan), np.full(len(y), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        if kernel == "precomputed":
            m = SVC(kernel="precomputed", probability=True, random_state=0, **svc_kw).fit(G_or_F[np.ix_(tr, tr)], y[tr])
            oof[va] = m.predict_proba(G_or_F[np.ix_(va, tr)])[:, 1]
            oofs[va] = m.decision_function(G_or_F[np.ix_(va, tr)])
        else:
            m = SVC(kernel=kernel, probability=True, random_state=0, **svc_kw).fit(G_or_F[tr], y[tr])
            oof[va] = m.predict_proba(G_or_F[va])[:, 1]
            oofs[va] = m.decision_function(G_or_F[va])
    return oof, oofs


def svm_tail(kind, Tr, ytr, Te, multilabel):
    """The as-run classical tail of the kernel cells (SVC with default settings, precomputed kernel
    for FQK, RBF on the standardised features for PQK), per label, plus out-of-fold training
    predictions. Returns (label, p_test, p_oof, s_test, s_oof): Platt probabilities as executed
    (used for calibration) and SVM decision scores (used for ranking).
    Platt scaling is fitted on an internal 5-fold split of the training data and with few samples
    it can fit a sigmoid of the wrong sign, reversing the ranking of the decision function,
    so ROC-AUC is computed from the decision scores."""
    kernel = "precomputed" if kind == "fqk" else "rbf"
    labels = range(ytr.shape[1]) if multilabel else [None]
    out = []
    for c in labels:
        yc = ytr[:, c] if c is not None else ytr
        if len(np.unique(yc)) < 2:
            continue
        svm = SVC(kernel=kernel, probability=True, random_state=0).fit(Tr, yc)
        p_oof, s_oof = _oof_proba(Tr, yc, kernel)
        out.append((c, svm.predict_proba(Te)[:, 1], p_oof, svm.decision_function(Te), s_oof))
    return out


# ----------------------------------------------------------------------------- replay
def _job(tag, jid):
    return load_archived(HW_DIR / tag / f"{jid}.json")


def _p0_concat(rec, nq):
    return np.concatenate([_counts_p0(r, nq) for r in rec["result"]])


def _pv_concat_sampler(rec):
    return np.concatenate([np.asarray(p[1]) for p in rec["inputs"]["pubs"]])


def _estimator_concat(rec, nq):
    pvs, evs = [], []
    for pub, res in zip(rec["inputs"]["pubs"], rec["result"]):
        pv = np.asarray(pub[2]); ev = np.asarray(res.data.evs)
        if pv.ndim == 3:                       # PQK: (1, n, nq) with (3N, 1) observables
            pvs.append(pv[0]); evs.append(ev.reshape(ev.shape[0], -1))
        else:                                  # VQC: (n, nq) with one observable
            pvs.append(pv.reshape(-1, nq)); evs.append(ev.reshape(1, -1))
    return np.concatenate(pvs, 0), np.concatenate(evs, 1)


def replay_cell(tag, cid, c, mdir):
    task, model, mitig = c["task"], c["model"], c["mitig"]
    nq = TASKS[task]["qubits"]; ml = TASKS[task]["type"] == "multilabel"
    d = cell_data(task, c["n_train"], c["n_test"])
    sub = np.zeros(len(d["ypool"]), bool); sub[np.asarray(c["sub_idx"])] = True
    ytr, ypool = d["ytr"], d["ypool"]
    jobs = {j["role"]: _job(tag, j["job"]) for j in c["jobs"]}
    align = {}
    if model == "fqk":
        recs = [jobs["LL"], jobs["trL"], jobs["teL"]]
        circ = recs[0]["inputs"]["pubs"][0][0]; order = [p.name for p in circ.parameters]
        a = [order.index(f"x[{k}]") for k in range(nq)]; b = [order.index(f"y[{k}]") for k in range(nq)]
        pvs = [_pv_concat_sampler(r) for r in recs]; ks = [_p0_concat(r, nq) for r in recs]
        L = int(round(np.sqrt(len(ks[0]))))
        land = pvs[0][::L][:, a]; K_LL = ks[0].reshape(L, L)
        hXtr = pvs[1][::L][:, a]; K_trL = ks[1].reshape(-1, L)
        hXte = pvs[2][::L][:, a]; K_teL = ks[2].reshape(-1, L)
        assert np.allclose(pvs[1][:L][:, b], land)
        align = {"train": float(np.abs(hXtr - d["Xtr_s"]).max()), "test": float(np.abs(hXte - d["Xpool_s"]).max())}
        w, V = np.linalg.eigh((K_LL + K_LL.T) / 2.0); keep = w > 1e-10
        Wh = (V[:, keep] * (1.0 / np.sqrt(w[keep]))) @ V[:, keep].T
        Phi_tr, Phi_te = K_trL @ Wh, K_teL @ Wh
        Tr, Te = Phi_tr @ Phi_tr.T, Phi_te @ Phi_tr.T
        np.savez_compressed(mdir / f"{task}_fqk_{mitig}.npz", K_LL=K_LL, K_trL=K_trL, K_teL=K_teL,
                            landmarks=land, Xtr_s=hXtr, Xte_s=hXte, G_tr=Tr, G_te=Te, in_sub=sub)
        preds = svm_tail("fqk", Tr, ytr, Te, ml)
    elif model == "pqk":
        hXtr, ev_tr = _estimator_concat(jobs["tr"], nq); hXte, ev_te = _estimator_concat(jobs["te"], nq)
        align = {"train": float(np.abs(hXtr - d["Xtr_s"]).max()), "test": float(np.abs(hXte - d["Xpool_s"]).max())}
        Ftr, Fte = ev_tr.T, ev_te.T                                    # (n, 3N) X,Y,Z per qubit
        s = StandardScaler().fit(Ftr)
        stem = "pqk" if c["encoding"] == "angle" else f"pqk-{c['encoding']}"
        np.savez_compressed(mdir / f"{task}_{stem}_{mitig}.npz", F_tr=Ftr, F_te=Fte, Xtr_s=hXtr, Xte_s=hXte, in_sub=sub)
        preds = svm_tail("pqk", s.transform(Ftr), ytr, s.transform(Fte), ml)
    else:
        labels = [c_ for c_ in range(ytr.shape[1]) if len(np.unique(ytr[:, c_])) > 1] if ml else [None]
        preds, evs_all, md = [], [], 0.0
        for lab in labels:
            hX, ev = _estimator_concat(jobs[f"label{lab if lab is not None else 0}"], nq)
            md = max(md, float(np.abs(hX - d["Xpool_s"]).max()))
            evs_all.append(ev.ravel()); pv_ = (1.0 + ev.ravel()) / 2.0
            preds.append((lab, pv_, None, pv_, None))
        align = {"test": md}
        vname = f"vqc-{c['vqc_train']}{'-noent' if c.get('ablate') else ''}"
        np.savez_compressed(mdir / f"{task}_{vname}_{mitig}.npz", Xte_s=hX, evs=np.vstack(evs_all),
                            in_sub=sub, labels=np.array([-1 if l is None else l for l in labels]))
    name = model if model != "vqc" else f"vqc-{c['vqc_train']}{'-noent' if c.get('ablate') else ''}"
    if model != "vqc" and c["encoding"] != "angle":
        name = f"{model}-{c['encoding']}"
    rows, a_sub, a_pool, a_logged = [], [], [], []
    for lab, p_te, p_oof, s_te, s_oof in preds:
        yc = ypool[:, lab] if lab is not None else ypool
        a_pool.append(roc_auc_score(yc, s_te)); a_sub.append(roc_auc_score(yc[sub], s_te[sub]))
        a_logged.append(roc_auc_score(yc[sub], p_te[sub]))
        for i in range(len(p_te)):
            rows.append(dict(split="test", idx=i, in_sub=bool(sub[i]), group=d["gpool"][i],
                             label=-1 if lab is None else lab, y_true=int(yc[i]), proba=float(p_te[i]),
                             score=float(s_te[i])))
        if p_oof is not None:
            yt = ytr[:, lab] if lab is not None else ytr
            for i in range(len(p_oof)):
                rows.append(dict(split="train_oof", idx=i, in_sub=True, group=d["gtr"][i],
                                 label=-1 if lab is None else lab, y_true=int(yt[i]), proba=float(p_oof[i]),
                                 score=float(s_oof[i])))
    for r in rows:
        r.update(tag=tag, task=task, model=name, encoding=c["encoding"], mitig=mitig)
    summ = dict(tag=tag, task=task, model=name, encoding=c["encoding"], mitig=mitig,
                roc_auc=float(np.mean(a_sub)), roc_auc_pool=float(np.mean(a_pool)),
                roc_auc_logged=float(np.mean(a_logged)),
                n_train=len(ytr), n_test=int(sub.sum()), n_pool=len(ypool), circuits=c.get("circuits"),
                shots=1024, quantum_seconds=c.get("quantum_seconds"),
                qpu_min=round((c.get("quantum_seconds") or 0) / 60.0, 2), session=c.get("session"),
                jobs=";".join(j["job"] for j in c["jobs"]), two_qubit=c.get("two_qubit"), depth=c.get("depth"),
                align_max_abs_diff=max(align.values()), backend=jobs[next(iter(jobs))]["backend"],
                per_class_auc=(";".join(f"{x:.4f}" for x in a_sub) if ml else None))
    return summ, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    plan = json.loads((HW_DIR / a.tag / "plan.json").read_text())
    out = OUT_ROOT / a.tag; mdir = out / "matrices"; mdir.mkdir(parents=True, exist_ok=True)
    summ, rows = [], []
    bell = plan.get("bell", [])
    for cid, c in plan["cells"].items():
        if c.get("status") != "DONE":
            continue
        s, r = replay_cell(a.tag, cid, c, mdir)
        s["bell"] = next((b["value"] for b in bell if b["session"] == c.get("session")), None)
        summ.append(s); rows += r
        print(f"{cid:26s} AUC sub {s['roc_auc']:.4f}  pool {s['roc_auc_pool']:.4f}  n={s['n_test']}/{s['n_pool']} "
              f"QPU {s['qpu_min']} min  align {s['align_max_abs_diff']:.1e}", flush=True)
    pd.DataFrame(summ).to_csv(out / "cells.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
