"""Pre-train the VQC weights (binary cross-entropy objective) for an hw_run2 hardware run (no QPU).

Trains with experiments/hw_run2.train_vqc (same data, seed, and file names), so hw_run2 later
loads the cached weights instead of training inside an open session.
  --train sv     noiseless statevector training
  --train noisy  noise-aware training against a device snapshot (control; training in
                 device-noise simulation)

Usage:
  uv run python experiments/train_vqc_hw.py --task apnea --encoding angle --train noisy \
      --noise ibm_phoenix_2026-09-25 --tag phoenix_2026-09
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments.config import HW2_PLAN                        # noqa: E402
from experiments.hw_run2 import cell_data, train_vqc, HW_DIR   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--train", choices=["sv", "noisy"], required=True)
    ap.add_argument("--noise", default="ibm_phoenix_2026-09-25")
    ap.add_argument("--tag", default="phoenix_2026-09")
    ap.add_argument("--labels", nargs="*", type=int, default=None, help="train only these label indices")
    ap.add_argument("--ablate", action="store_true", help="entanglement-removed control")
    a = ap.parse_args()
    c = next(x for x in HW2_PLAN if x["task"] == a.task and x["model"] == "vqc")
    d = cell_data(a.task, c["n_train"], c["n_test"])
    out = HW_DIR / a.tag; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    w = train_vqc(a.task, a.encoding, d, a.train, a.noise, out, only_labels=a.labels, ablate=a.ablate)
    print(f"trained {a.task} vqc/{a.encoding} ({a.train}) labels={list(w)} in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
