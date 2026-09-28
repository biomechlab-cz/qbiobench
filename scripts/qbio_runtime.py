"""QBioBench hardware/reproducibility runtime helpers.

Two groups:
  * Pure / safe: config_hash, run-manifest writer, budget ledger.
  * Hardware-touching: get_service (pinned IBM instance), snapshot_backend, select_layout
    (mapomatic, not used by the executed runs), and calibration_probe_ok, a stub that was never
    implemented (the runner applies an absolute Bell threshold itself, see experiments/hw_run2.py).

No hardware call happens on import.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UV_LOCK = PROJECT_ROOT / "uv.lock"

PRIMARY_BACKEND = "ibm_phoenix"    # Nighthawk r2
SECONDARY_BACKEND = "ibm_fez"      # Heron r2


# --- Reproducibility (pure) -------------------------------------------------

def _source_tree_hash() -> str:
    """SHA-256 over the project's source (.py under experiments/, utils/, scripts/),
    sorted for determinism. Stands in for a commit SHA so that any code change changes the
    hash. Excludes caches and the local credential scripts (scripts/ibm_init*.py), which are
    not part of the released code."""
    h = hashlib.sha256()
    files = []
    for sub in ("experiments", "utils", "scripts"):
        files += sorted((PROJECT_ROOT / sub).rglob("*.py"))
    for f in files:
        rel = f.relative_to(PROJECT_ROOT)
        if "__pycache__" in f.parts or (rel.parent.as_posix() == "scripts"
                                        and fnmatch.fnmatch(rel.name, "ibm_init*.py")):
            continue
        h.update(rel.as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def config_hash(config: dict) -> str:
    """SHA-256 over uv.lock + the source tree + the canonical JSON of `config`.
    Stands in for a commit hash in run manifests."""
    h = hashlib.sha256()
    if UV_LOCK.exists():
        h.update(UV_LOCK.read_bytes())
    h.update(_source_tree_hash().encode())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def write_manifest(name: str, config: dict, results: dict, *, backend: str | None = None,
                   calibration: dict | None = None, subdir: str = "manifests") -> Path:
    """Write a run manifest JSON next to outputs in ./data. Every result row
    must be reconstructable from this."""
    out_dir = DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "name": name,
        "utc": stamp,
        "config_hash": config_hash(config),
        "config": config,
        "backend": backend,
        "calibration": calibration,
        "results": results,
    }
    path = out_dir / f"{name}_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def log_budget(date: str, backend: str, purpose: str, minutes: float):
    """Append a line to the IBM budget ledger and return cumulative minutes this
    month. The ledger is a local record of QPU use and is not part of the public release."""
    ledger = DATA_DIR / "ibm_budget_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {"date": date, "backend": backend, "purpose": purpose, "minutes": minutes}
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    month = date[:7]
    total = 0.0
    for line in ledger.read_text().splitlines():
        e = json.loads(line)
        if e["date"].startswith(month):
            total += e["minutes"]
    return total


# --- Hardware ----------------------------------------------------------------

DEFAULT_INSTANCE = "Research_CVUT"   # us-east premium instance (ibm_phoenix, ...)


def get_service(instance: str | None = None):
    """Return a QiskitRuntimeService pinned to one IBM Quantum instance.

    The saved default account may resolve to a different region (the EU instance exposes
    only ibm_aachen/ibm_berlin, and us-east jobs then report RuntimeJobNotFound). The
    instance is therefore always pinned: `instance` (name or CRN), else $QBIO_IBM_INSTANCE,
    else DEFAULT_INSTANCE. Credentials come from the saved account (scripts/ibm_login.py) or
    from $IBM_QUANTUM_TOKEN. Import is local so nothing authenticates on module import."""
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService
    want = instance or os.environ.get("QBIO_IBM_INSTANCE") or DEFAULT_INSTANCE
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    base = (QiskitRuntimeService(channel="ibm_quantum_platform", token=tok) if tok
            else QiskitRuntimeService())
    if want.startswith("crn:"):
        crn = want
    else:
        crn = next((i["crn"] for i in base.instances() if i.get("name") == want), None)
        if crn is None:
            raise RuntimeError(f"IBM instance {want!r} not visible to this account")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=crn)
    return QiskitRuntimeService(channel="ibm_quantum_platform", instance=crn)


def snapshot_backend(backend) -> dict:
    """Capture backend.properties()/configuration into a JSON-serializable dict
    (median T1/T2, gate errors, basis gates, coupling map, revision, calib date).
    Metadata only — no circuits executed, no QPU time spent."""
    import numpy as np
    props = backend.properties()
    cfg = backend.configuration()

    def _safe(f, q):                     # faulty qubits can lack T1/T2/readout entries
        try:
            return f(q)
        except Exception:
            return None
    t1 = [_safe(props.t1, q) for q in range(backend.num_qubits)]
    t2 = [_safe(props.t2, q) for q in range(backend.num_qubits)]
    # median 2q (cz/ecr) gate error across the device
    cz_errs = []
    for g in props.gates:
        if g.gate in ("cz", "ecr", "cx") and len(g.qubits) == 2:
            for p in g.parameters:
                if p.name == "gate_error":
                    cz_errs.append(p.value)
    readout = [_safe(props.readout_error, q) for q in range(backend.num_qubits)]
    snap = {
        "backend": backend.name,
        "n_qubits": backend.num_qubits,
        "basis_gates": list(cfg.basis_gates),
        "processor": getattr(cfg, "processor_type", None),
        "calibration_date": str(props.last_update_date),
        "median_t1_us": float(np.median([x for x in t1 if x]) * 1e6),
        "median_t2_us": float(np.median([x for x in t2 if x]) * 1e6),
        "median_2q_error": float(np.median(cz_errs)) if cz_errs else None,
        "median_readout_error": float(np.median([x for x in readout if x is not None])),
        "n_edges": len(list(backend.coupling_map)) if backend.coupling_map else None,
    }
    return snap


def calibration_probe_ok(backend, *, sigma: float = 2.0) -> bool:
    """Bell-pair fidelity probe; reject Sessions >sigma from the day median of the Bell-pair
    fidelity P(00)+P(11). Not implemented: the executed runs instead abort a session whose
    transpiled Bell pair gives P(00)+P(11) < 0.90 (4096 shots, see experiments/hw_run2.py).
    Returns True if the Session is acceptable."""
    raise NotImplementedError("Phase 3: Bell-pair fidelity calibration gate")


def select_layout(circuit, backend):
    """mapomatic best-layout selection for a transpiled circuit. Returns the
    chosen layout + its estimated error. Metadata only — no QPU time."""
    import mapomatic as mm
    small = mm.deflate_circuit(circuit)
    layouts = mm.matching_layouts(small, backend)
    scored = mm.evaluate_layouts(small, layouts, backend)
    if not scored:
        return {"layout": None, "score": None}
    best, score = scored[0]
    return {"layout": list(best), "score": float(score)}
