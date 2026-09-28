"""Build a device noise model + transpiler target snapshot for noisy Aer simulation.

Sources (metadata only, no QPU time):
  --from-job JOB.json   backend properties stored in an archived job record (the device state at
                        execution time)
  --live BACKEND        current properties of a live backend (e.g. ibm_phoenix)

Output data/noise/<name>.pkl holds {noise_model (dict), target (qiskit Target), properties
(dict), configuration (dict), source}. The target carries per-qubit gate and readout errors, so
the preset pass manager (VF2 layout scoring) places simulated circuits on low-error qubits the
same way transpilation does for hardware, and the noise model then attaches the matching errors.

Shipped snapshots: heron_r3_2026-06-11 (noise snapshot of an IBM Heron r3 processor, June 2026,
the default of the simulation screen) and ibm_phoenix_2026-09-25 (live IBM Phoenix properties).

Usage:
  uv run python scripts/build_noise_model.py --from-job data/hardware/<tag>/<job_id>.json --name <name>
  uv run python scripts/build_noise_model.py --live ibm_phoenix --name ibm_phoenix_2026-09-25
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import qbio_runtime as rt  # noqa: E402

OUT = ROOT / "data" / "noise"


def _with_frequency(props_obj):
    """Aer's from_backend_properties reads a per-qubit 'frequency' that current IBM backends
    no longer report. At temperature 0 (the default) it only enters the excited-state
    population, which is then exactly zero, so a placeholder value has no effect."""
    import numpy as np
    from qiskit_ibm_runtime.models import BackendProperties
    d = props_obj.to_dict()
    # qubits flagged faulty can lack T1/T2; fill with device medians (the target from
    # convert_to_target(filter_faulty=True) keeps the transpiler off them anyway)
    med = {n: float(np.median([p["value"] for q in d["qubits"] for p in q if p["name"] == n]))
           for n in ("T1", "T2")}
    for q in d["qubits"]:
        have = {p["name"] for p in q}
        for n, unit in (("T1", "us"), ("T2", "us")):
            if n not in have:
                q.append({"date": d["last_update_date"], "name": n, "unit": unit, "value": med[n]})
        if "frequency" not in have:
            q.append({"date": d["last_update_date"], "name": "frequency", "unit": "GHz", "value": 5.0})
    return BackendProperties.from_dict(d)


def build(props_obj, configuration, source):
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    nm = NoiseModel.from_backend_properties(_with_frequency(props_obj), temperature=0)
    target = convert_to_target(configuration, props_obj, include_control_flow=False,
                               include_fractional_gates=False)
    return {"noise_model": nm.to_dict(serializable=False), "target": target,
            "properties": props_obj.to_dict(), "configuration": configuration.to_dict(),
            "source": source, "basis_gates": nm.basis_gates}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-job")
    g.add_argument("--live")
    ap.add_argument("--name", required=True)
    ap.add_argument("--instance", default=None)
    a = ap.parse_args()
    from qiskit_ibm_runtime.models import BackendProperties
    svc = rt.get_service(a.instance)
    if a.from_job:
        from qiskit_ibm_runtime.utils import RuntimeDecoder
        rec = json.loads(Path(a.from_job).read_text())          # keep raw dict for properties
        props = BackendProperties.from_dict(json.loads(json.dumps(rec["backend_properties"]),
                                                       cls=RuntimeDecoder))
        cfg = svc.backend(rec["backend"]).configuration()       # coupling map / basis (static)
        src = f"job {rec['job_id']} ({rec['backend']}, properties {props.last_update_date})"
    else:
        b = svc.backend(a.live)
        props, cfg = b.properties(), b.configuration()
        src = f"live {a.live} properties {props.last_update_date}"
    snap = build(props, cfg, src)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(snap, f)
    print(f"saved {path.relative_to(ROOT)}  ({src}; basis {snap['basis_gates']})")


if __name__ == "__main__":
    main()
