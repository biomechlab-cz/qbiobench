"""IBM Phoenix hardware runner: full-pool evaluation on ibm_phoenix (exploratory, post hoc).

What it runs (config.HW2_PLAN):
  * the kernel cells with the fixed protocol of config.HW_PLAN (training sizes, 16 landmarks,
    1024 shots, XY4 dynamical decoupling, mitigation none / TREX, optimization_level 3, seed 0),
    with the test kernel / features measured on the FULL capped test pool. The fixed test
    subsets (80 FQK, 416 PQK units, recorded as `sub_idx`) lie inside that pool, so the subset
    metric and the full-pool evaluation come from the same measurements and the same model,
    trained on the kernels / features measured on this device.
  * the VQC with the binary cross-entropy objective (models.py), weights trained in
    simulation before the session opens: noiseless ('sv') and noise-aware against the device
    snapshot ('noisy'), both evaluated on hardware.

Everything the replay needs is written to data/hardware/<tag>/plan.json (cell, role, job id,
sample indices), the raw jobs are archived with experiments/hw_archive.py, and
experiments/hw_replay2.py recomputes all metrics from those records. Predictions are therefore
stored natively for every cell.

Budget: a guard stops before a cell if the logged QPU minutes of this run exceed --budget-min.

Usage:
  uv run python experiments/hw_run2.py --backend ibm_phoenix --tag phoenix_2026-09 --dry-run
  uv run python experiments/hw_run2.py --backend ibm_phoenix --tag phoenix_2026-09 --budget-min 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments.config import TASKS, HW2_PLAN, HW_SHOTS as SHOTS, RESILIENCE      # noqa: E402
from experiments.models import feature_map, _pauli, _parity_obs, ANGLE_RANGE, VQCReuploading  # noqa: E402
from experiments.screen import load_task_split                                       # noqa: E402
import qbio_runtime as rt                                                            # noqa: E402

HW_DIR = ROOT / "data" / "hardware"


# ----------------------------------------------------------------------------- data
def cell_data(task, n_train, n_test):
    """Train subsample and the full test pool, plus the indices of the fixed test subsample
    (the seeded n_test draw) inside that pool. Reproduces load_task_split's seeded draws exactly."""
    Xtr, ytr, Xte_sub, yte_sub, nq, ml, gtr, gte_sub = load_task_split(task, n_train, n_test, 0,
                                                                        return_groups=True)
    _, _, Xpool, ypool, _, _, _, gpool = load_task_split(task, n_train, None, 0, return_groups=True)
    # locate the subsample rows inside the pool (exact feature match)
    idx = np.array([int(np.argmin(np.abs(Xpool - x).sum(1))) for x in Xte_sub])
    assert np.allclose(Xpool[idx], Xte_sub) and len(set(idx)) == len(idx)
    sc = MinMaxScaler(ANGLE_RANGE).fit(np.asarray(Xtr, float)[:, :nq])
    return dict(Xtr=Xtr, ytr=ytr, gtr=gtr, Xpool=Xpool, ypool=ypool, gpool=gpool, sub_idx=idx,
                Xtr_s=sc.transform(Xtr[:, :nq]), Xpool_s=sc.transform(Xpool[:, :nq]), nq=nq, ml=ml)


# ----------------------------------------------------------------------------- PUB builders
def isa(circ, backend):
    return transpile(circ, backend=backend, optimization_level=3, seed_transpiler=0)


def fqk_pubs(d, L, backend):
    nq = d["nq"]; fm = feature_map("angle", nq)
    rng = np.random.default_rng(0)
    lidx = rng.choice(len(d["Xtr_s"]), L, replace=False)          # seeded landmark draw of the kernel protocol
    land = d["Xtr_s"][lidx]
    pb = ParameterVector("y", nq)
    inv = fm.inverse().assign_parameters({p: pb[i] for i, p in enumerate(fm.parameters)})
    t = QuantumCircuit(nq, nq); t.compose(fm, inplace=True); t.compose(inv, inplace=True)
    t.measure(range(nq), range(nq))
    it = isa(t, backend); order = [p.name for p in it.parameters]
    a = [order.index(f"x[{k}]") for k in range(nq)]; b = [order.index(f"y[{k}]") for k in range(nq)]

    def block(XA, XB):
        ai = np.repeat(np.arange(len(XA)), len(XB)); bi = np.tile(np.arange(len(XB)), len(XA))
        pv = np.zeros((len(ai), len(order))); pv[:, a] = XA[ai]; pv[:, b] = XB[bi]
        return (it, pv)
    pubs = [("LL", block(land, land)), ("trL", block(d["Xtr_s"], land)), ("teL", block(d["Xpool_s"], land))]
    return pubs, {"landmark_idx": lidx.tolist(), "two_qubit": it.num_nonlocal_gates(), "depth": it.depth()}


def pqk_pubs(d, backend, encoding="angle"):
    nq = d["nq"]; fm = feature_map(encoding, nq); it = isa(fm, backend)
    obs = [_pauli(Lt, i, nq).apply_layout(it.layout) for i in range(nq) for Lt in "XYZ"]
    arr = np.empty((len(obs), 1), dtype=object)
    for i, o in enumerate(obs):
        arr[i, 0] = o
    pubs = [("tr", (it, arr, d["Xtr_s"][np.newaxis])), ("te", (it, arr, d["Xpool_s"][np.newaxis]))]
    return pubs, {"two_qubit": it.num_nonlocal_gates(), "depth": it.depth()}


def vqc_pubs(d, enc, weights_by_label, backend):
    """One PUB per label: circuit with trained weights bound, inputs = full test pool."""
    nq = d["nq"]; pubs = []; meta = {}
    for lab, (vqc, w) in weights_by_label.items():
        bound = vqc.circ.assign_parameters({vqc.w_p[i]: w[i] for i in range(len(w))})
        it = isa(bound, backend)
        ob = _parity_obs(nq).apply_layout(it.layout)
        order = [p.name for p in it.parameters]
        cols = [order.index(f"x[{k}]") for k in range(nq)]
        pv = np.zeros((len(d["Xpool_s"]), nq)); pv[:, cols] = d["Xpool_s"]
        pubs.append((f"label{lab}", (it, [ob], pv)))
        meta = {"two_qubit": it.num_nonlocal_gates(), "depth": it.depth()}
    return pubs, meta


# ----------------------------------------------------------------------------- VQC weights
def train_vqc(task, enc, d, train_mode, noise, out_dir, only_labels=None, ablate=False):
    """Train (or load cached) VQC weights (binary cross-entropy objective) for every label of a task."""
    import os
    res = {}
    labels = [c for c in range(d["ytr"].shape[1]) if len(np.unique(d["ytr"][:, c])) > 1] if d["ml"] else [None]
    if only_labels is not None:
        labels = [c for c in labels if (c if c is not None else 0) in only_labels]
    for c in labels:
        tag = f"{train_mode}{'_noent' if ablate else ''}"
        f = out_dir / f"vqc_{task}_{enc}_{tag}_label{c if c is not None else 0}.npz"
        yc = d["ytr"][:, c] if c is not None else d["ytr"]
        if train_mode == "noisy":
            os.environ["QBIO_NOISE"] = noise
        v = VQCReuploading(enc, d["nq"], mode="sv", train_mode=train_mode, seed=0, record_every=20,
                           ablate_ent=ablate)
        if f.exists():
            w = np.load(f)["weights"]
        else:
            v.fit(d["Xtr"], yc)
            w = v.weights
            np.savez(f, weights=w, history=json.dumps(v.history), fit=json.dumps(v.fit_result))
        res[c if c is not None else 0] = (v, w)
    return res


# ----------------------------------------------------------------------------- execution
MAX_SETS = 4096     # parameter sets per PUB; larger blocks are split into several PUBs of one job


def _chunk(pub):
    """Split a PUB along its parameter-set axis (the replay concatenates the PUB results)."""
    if len(pub) == 2:                                   # sampler (circ, pv[m, P])
        circ, pv = pub
        return [(circ, pv[i:i + MAX_SETS]) for i in range(0, len(pv), MAX_SETS)]
    circ, obs, pv = pub                                 # estimator
    if pv.ndim == 3:                                    # (1, n, P): PQK broadcast over observables
        return [(circ, obs, pv[:, i:i + MAX_SETS]) for i in range(0, pv.shape[1], MAX_SETS)]
    return [(circ, obs, pv[i:i + MAX_SETS]) for i in range(0, len(pv), MAX_SETS)]


def make_primitives(session, resilience):
    from qiskit_ibm_runtime import SamplerV2, EstimatorV2
    est = EstimatorV2(mode=session); smp = SamplerV2(mode=session)
    for p in (est, smp):
        p.options.dynamical_decoupling.enable = True
        p.options.dynamical_decoupling.sequence_type = "XY4"
        p.options.default_shots = SHOTS
    est.options.resilience_level = resilience            # 0 none, 1 TREX (twirled readout)
    smp.options.twirling.enable_measure = resilience >= 1  # Sampler: measurement twirling only
    return est, smp


def bell_probe(backend, session):
    from qiskit_ibm_runtime import SamplerV2
    qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    job = SamplerV2(mode=session).run([isa(qc, backend)], shots=4096)
    c = job.result()[0].data.c.get_counts(); tot = sum(c.values())
    return (c.get("00", 0) + c.get("11", 0)) / tot, job.job_id()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_phoenix")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--budget-min", type=float, default=90.0)
    ap.add_argument("--only", nargs="*", default=None, help="cell ids to run (default: all in order)")
    ap.add_argument("--dry-run", action="store_true", help="build PUBs and print costs; no QPU")
    ap.add_argument("--probe", action="store_true", help="cost probe: Bell + one FQK block, then stop")
    ap.add_argument("--noise", default=None, help="noise snapshot for noise-aware VQC training")
    a = ap.parse_args()
    out = HW_DIR / a.tag; out.mkdir(parents=True, exist_ok=True)
    svc = rt.get_service(a.instance); backend = svc.backend(a.backend)
    snap = rt.snapshot_backend(backend)
    print(f"{a.backend} {snap['processor']} calib {snap['calibration_date']} "
          f"2Q {snap['median_2q_error']:.2e} RO {snap['median_readout_error']:.2e} "
          f"queue {backend.status().pending_jobs}", flush=True)
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else \
        {"backend": a.backend, "tag": a.tag, "shots": SHOTS, "snapshot": snap, "cells": {}}
    cells = [c for c in HW2_PLAN if (a.only is None or c["id"] in a.only)]

    # ---- build every PUB first (classical work, no QPU) ----
    built = {}
    for c in cells:
        d = cell_data(c["task"], c["n_train"], c["n_test"])
        if c["model"] == "fqk":
            pubs, meta = fqk_pubs(d, c["L"], backend)
        elif c["model"] == "pqk":
            pubs, meta = pqk_pubs(d, backend, c["encoding"])
        else:
            noise = a.noise or f"{a.backend}_{snap['calibration_date'][:10]}"
            tagv = f"{c['vqc_train']}{'_noent' if c.get('ablate') else ''}"
            if a.dry_run and not list(out.glob(f"vqc_{c['task']}_{c['encoding']}_{tagv}_label*.npz")):
                v = VQCReuploading(c["encoding"], d["nq"], seed=0)       # structure only
                labs = range(d["ytr"].shape[1]) if d["ml"] else [0]
                w = {lab: (v, np.zeros(v.n_weights)) for lab in labs}
            else:
                w = train_vqc(c["task"], c["encoding"], d, c["vqc_train"], noise, out, ablate=c.get("ablate", False))
            pubs, meta = vqc_pubs(d, c["encoding"], w, backend)
        n_circ = sum(int(np.prod(np.asarray(p[1][-1]).shape[:-1])) for p in pubs)
        built[c["id"]] = (c, d, pubs, meta, n_circ)
        print(f"  built {c['id']:28s} pubs={len(pubs)} param-sets={n_circ:6d} 2q={meta.get('two_qubit')} "
              f"depth={meta.get('depth')} ntr={len(d['ytr'])} pool={len(d['ypool'])} sub={len(d['sub_idx'])}",
              flush=True)
    if a.dry_run:
        print("dry run: nothing submitted"); return

    from qiskit_ibm_runtime import Session
    if a.probe:
        # cost probe: Bell pair + the first FQK landmark block (256 circuits, 1024 shots); project
        # the whole plan from the measured QPU seconds per circuit. Results kept as provenance.
        cid = next(k for k, v in built.items() if v[0]["model"] == "fqk")
        _, _, pubs, _, _ = built[cid]
        with Session(backend=backend, max_time=1800) as session:
            fid, bjob = bell_probe(backend, session)
            est, smp = make_primitives(session, 0)
            j = smp.run([pubs[0][1]]); j.result()
            qs = (svc.job(j.job_id()).metrics().get("usage") or {}).get("quantum_seconds") or 0
            qb = (svc.job(bjob).metrics().get("usage") or {}).get("quantum_seconds") or 0
        n0 = len(pubs[0][1][1])
        per = qs / n0
        total = sum(v[4] for v in built.values())
        plan.setdefault("probes", []).append({"bell": fid, "bell_job": bjob, "block_job": j.job_id(),
                                              "circuits": n0, "quantum_seconds": qs, "bell_seconds": qb,
                                              "utc": datetime.now(timezone.utc).isoformat()})
        plan_path.write_text(json.dumps(plan, indent=1, default=str))
        rt.log_budget(datetime.now(timezone.utc).strftime("%Y-%m-%d"), a.backend, f"{a.tag}/probe", (qs + qb) / 60.0)
        print(f"PROBE Bell {fid:.4f} ({qb}s) | {n0} circuits -> {qs} QPU-s = {per*1000:.1f} ms/circuit | "
              f"projected for {total} param-sets (x3 bases for PQK not included): {per*total/60:.1f} min")
        return
    used = sum(v.get("quantum_seconds", 0) for v in plan["cells"].values()) / 60.0
    session = Session(backend=backend, max_time=4 * 3600)
    try:
        _run_cells(a, svc, backend, session, built, plan, plan_path, used)
    finally:
        try:
            session.close()
        except Exception as e:                           # a failed close must not lose results
            print(f"(session close failed: {str(e)[:80]}; results already saved)")
    print("session closed")


def _run_cells(a, svc, backend, session, built, plan, plan_path, used):
    if True:
        fid, bjob = bell_probe(backend, session)
        plan.setdefault("bell", []).append({"value": fid, "job": bjob, "session": session.session_id,
                                            "utc": datetime.now(timezone.utc).isoformat()})
        print(f"session {session.session_id}  Bell P(00)+P(11) = {fid:.4f}", flush=True)
        plan_path.write_text(json.dumps(plan, indent=1, default=str))
        if fid < 0.90:
            print("ABORT: Bell probe below 0.90"); return
        for cid, (c, d, pubs, meta, n_circ) in built.items():
            if cid in plan["cells"] and plan["cells"][cid].get("status") == "DONE":
                print(f"  skip {cid} (done)"); continue
            if used >= a.budget_min:
                print(f"STOP budget: {used:.1f} >= {a.budget_min} min before {cid}"); break
            est, smp = make_primitives(session, RESILIENCE[c["mitig"]])
            prim = smp if c["model"] == "fqk" else est
            t0 = time.time(); jobs = []
            for role, pub in pubs:
                chunks = _chunk(pub)
                j = prim.run(chunks); jobs.append({"role": role, "job": j.job_id(), "n_pubs": len(chunks)})
                j.result()                                  # sequential: keeps the budget guard exact
            qs = sum(((svc.job(x["job"]).metrics().get("usage") or {}).get("quantum_seconds") or 0) for x in jobs)
            used += qs / 60.0
            plan["cells"][cid] = {**c, "jobs": jobs, "status": "DONE", "quantum_seconds": qs,
                                  "wall_s": round(time.time() - t0, 1), "session": session.session_id,
                                  "sub_idx": d["sub_idx"].tolist(), "circuits": n_circ, **meta}
            plan_path.write_text(json.dumps(plan, indent=1, default=str))
            rt.log_budget(datetime.now(timezone.utc).strftime("%Y-%m-%d"), a.backend, f"{a.tag}/{cid}", qs / 60.0)
            print(f"  DONE {cid:28s} {qs:5d} QPU-s ({qs/60:.2f} min; run total {used:.1f} min)  "
                  f"wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
