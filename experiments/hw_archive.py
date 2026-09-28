"""Archive IBM Quantum jobs locally (free cloud reads, no QPU time).

For every job this writes one JSON file holding the submitted PUBs (transpiled circuits and
parameter values), the primitive options, the full result (counts / expectation values and
execution spans), the usage metrics, and the backend properties at execution time. The archive
is the raw hardware record that every downstream hardware number is recomputed from
(experiments/hw_replay2.py), so nothing depends on run-time bookkeeping.

Usage:
  uv run python experiments/hw_archive.py --tag phoenix_2026-09 --jobs <job id> <job id> ...
  uv run python experiments/hw_archive.py --tag <tag> --backend ibm_phoenix --after 2026-09-25 --before 2026-09-27
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import qbio_runtime as rt  # noqa: E402

HW_DIR = ROOT / "data" / "hardware"


def _props_dict(job):
    try:
        p = job.properties()
        return p.to_dict() if p is not None else None
    except Exception as e:                      # properties are optional provenance
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def archive_job(job, out_dir: Path, *, overwrite=False) -> Path:
    from qiskit_ibm_runtime.utils import RuntimeEncoder
    path = out_dir / f"{job.job_id()}.json"
    if path.exists() and not overwrite:
        return path
    status = str(job.status())
    rec = {
        "job_id": job.job_id(),
        "session_id": job.session_id,
        "backend": job.backend().name,
        "primitive": job.primitive_id,
        "status": status,
        "creation_date": str(job.creation_date),
        "metrics": job.metrics(),
        "inputs": job.inputs,
        "result": job.result() if status == "DONE" else None,
        "backend_properties": _props_dict(job),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, cls=RuntimeEncoder))
    return path


def load_archived(path: Path) -> dict:
    """Inverse of archive_job: decode PUBs, results, and arrays back to Qiskit objects.
    Accepts <job>.json or the gzip-compressed <job>.json.gz of the public release."""
    import gzip
    from qiskit_ibm_runtime.utils import RuntimeDecoder
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    text = gzip.open(path, "rt", encoding="utf8").read() if path.suffix == ".gz" else path.read_text()
    return json.loads(text, cls=RuntimeDecoder)


def archived_jobs(directory: Path) -> list[Path]:
    """All job records in an archive directory (.json or .json.gz), excluding the archive
    index (index.json) and the run plan written by hw_run2.py (plan.json)."""
    files = [p for p in Path(directory).glob("*.json") if p.name not in ("index.json", "plan.json")]
    files += [p for p in Path(directory).glob("*.json.gz")]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="subdirectory under data/hardware/")
    ap.add_argument("--backend")
    ap.add_argument("--after", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--before", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--jobs", nargs="*")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    svc = rt.get_service(a.instance)
    if a.jobs:
        jobs = [svc.job(j) for j in a.jobs]
    else:
        kw = {"limit": 1000, "descending": False}
        if a.backend:
            kw["backend_name"] = a.backend
        if a.after:
            kw["created_after"] = datetime.fromisoformat(a.after).replace(tzinfo=timezone.utc)
        if a.before:
            kw["created_before"] = datetime.fromisoformat(a.before).replace(tzinfo=timezone.utc)
        jobs = svc.jobs(**kw)
    out = HW_DIR / a.tag
    index = []
    for j in jobs:
        p = archive_job(j, out, overwrite=a.overwrite)
        m = j.metrics() or {}
        index.append({"job_id": j.job_id(), "session_id": j.session_id, "backend": j.backend().name,
                      "primitive": j.primitive_id, "status": str(j.status()),
                      "created": str(j.creation_date),
                      "quantum_seconds": (m.get("usage") or {}).get("quantum_seconds"),
                      "file": p.name})
        print(f"  {j.job_id()} {j.primitive_id:9s} {index[-1]['status']:9s} "
              f"{index[-1]['quantum_seconds']} s -> {p.name}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"archived {len(index)} jobs to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
