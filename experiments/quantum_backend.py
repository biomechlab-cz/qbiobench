"""Primitive factory for simulation (noiseless statevector and device-noise Aer).

Modes:
  sv     - Aer statevector, exact expectations, logical circuits (noiseless reference)
  noisy  - Aer with a device noise snapshot. Circuits MUST be transpiled to the device ISA
           first (use `sim_pass_manager()`): the Aer V2 primitives do not transpile, and a
           noise model only attaches errors to the gate names it defines (cz, sx, x, rz,
           measure, ...). Untranspiled `ry`/`cx`/`h`/`p` gates would run noiselessly and an
           estimator without `measure` would see no readout error. `NoisyCircuit` below
           handles the transpilation, compaction, and noise-model remapping.

The default snapshot, heron_r3_2026-06-11, is the noise snapshot of an IBM Heron r3 processor
(June 2026) used by the simulation screen (scripts/build_noise_model.py). Select another with
$QBIO_NOISE (file stem in data/noise/).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel

ROOT = Path(__file__).resolve().parent.parent
NOISE_DIR = ROOT / "data" / "noise"
DEFAULT_NOISE = "heron_r3_2026-06-11"      # noise snapshot of an IBM Heron r3 processor (June 2026)
_cache: dict = {}


def _threads() -> dict:
    """Aer thread cap for process-parallel campaigns ($QBIO_AER_THREADS, 0 = Aer default)."""
    n = int(os.environ.get("QBIO_AER_THREADS", "0"))
    return {"max_parallel_threads": n} if n > 0 else {}


def noise_name() -> str:
    return os.environ.get("QBIO_NOISE", DEFAULT_NOISE)


def load_snapshot(name: str | None = None) -> dict:
    name = name or noise_name()
    if name not in _cache:
        with open(NOISE_DIR / f"{name}.pkl", "rb") as f:
            snap = pickle.load(f)
        snap["nm"] = NoiseModel.from_dict(snap["noise_model"])
        _cache[name] = snap
    return _cache[name]


def load_noise_model(name: str | None = None) -> NoiseModel:
    return load_snapshot(name)["nm"]


def sim_target(name: str | None = None):
    return load_snapshot(name)["target"]


def sim_pass_manager(name: str | None = None, *, optimization_level: int = 3, seed: int = 0):
    """Preset pass manager onto the snapshot target (error-aware VF2 layout, device basis).
    The same settings as the hardware runner, so simulated and executed circuits match."""
    from qiskit.transpiler import generate_preset_pass_manager
    key = ("pm", name or noise_name(), optimization_level, seed)
    if key not in _cache:
        _cache[key] = generate_preset_pass_manager(optimization_level=optimization_level,
                                                   target=sim_target(name), seed_transpiler=seed)
    return _cache[key]


DM_MAX_QUBITS = 10                                        # exact density matrix up to here
N_TRAJECTORIES = int(os.environ.get("QBIO_TRAJ", "256"))  # noise trajectories above it


class NoisyCircuit:
    """A logical circuit compiled for device-faithful noisy simulation.

    The circuit is transpiled onto the snapshot target (error-aware layout, device basis) and
    then compacted onto its active physical qubits, with the noise model remapped to the same
    qubits. Compacting is required because the Aer EstimatorV2 saves expectation values over
    every circuit qubit (156 for the full device), which a density-matrix simulation cannot
    hold. Observables given on the logical qubits are mapped through the layout."""

    def __init__(self, circ, name: str | None = None, *, seed: int = 0):
        from qiskit import QuantumCircuit
        from qiskit_aer.noise import NoiseModel as _NM
        pm = sim_pass_manager(name, seed=seed)
        isa = pm.run(circ)
        self.isa = isa
        self.layout = isa.layout
        active = sorted({isa.find_bit(q).index for inst in isa.data for q in inst.qubits})
        self.phys = active
        idx = {p: i for i, p in enumerate(active)}
        n_c = sum(len(r) for r in isa.cregs)
        qc = QuantumCircuit(len(active), n_c) if n_c else QuantumCircuit(len(active))
        for inst in isa.data:
            qs = [qc.qubits[idx[isa.find_bit(q).index]] for q in inst.qubits]
            cs = [qc.clbits[isa.find_bit(c).index] for c in inst.clbits]
            qc.append(inst.operation, qs, cs)
        self.circuit = qc
        self.parameters = list(qc.parameters)
        # remap the noise model onto the compact register
        full = load_noise_model(name)
        nm = _NM(basis_gates=full.basis_gates)
        for gate, qerrs in full._local_quantum_errors.items():
            for qubits, err in qerrs.items():
                if all(q in idx for q in qubits):
                    nm.add_quantum_error(err, gate, [idx[q] for q in qubits], warnings=False)
        for qubits, rerr in full._local_readout_errors.items():
            if all(q in idx for q in qubits):
                nm.add_readout_error(rerr, [idx[q] for q in qubits], warnings=False)
        self.noise_model = nm
        self.two_qubit_gates = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)
        self.depth = isa.depth()

    def observable(self, logical_op):
        """Map a SparsePauliOp on the logical qubits onto the compact register."""
        from qiskit.quantum_info import SparsePauliOp
        full = logical_op.apply_layout(self.layout)
        labels = []
        for p in full.paulis:
            s = p.to_label()[::-1]                     # s[k] acts on physical qubit k
            for k, ch in enumerate(s):
                if ch != "I" and k not in self.phys:
                    raise ValueError(f"observable acts on idle physical qubit {k}")
            labels.append("".join(s[k] for k in self.phys)[::-1])
        return SparsePauliOp(labels, full.coeffs)

    @property
    def method(self):
        """Exact density matrix up to DM_MAX_QUBITS active qubits, else Monte-Carlo trajectories.
        8-qubit ring/circular entanglers route through 12 heavy-hex qubits, where a density
        matrix (4^12 entries per bind) is impractical."""
        return "density_matrix" if len(self.phys) <= DM_MAX_QUBITS else "statevector"

    def estimator(self, shots=4096, seed=0):
        run = {"seed_simulator": seed}
        if self.method == "statevector":
            # save_expectation_value averages over shots = noise trajectories
            run["shots"] = N_TRAJECTORIES
        return EstimatorV2(options={"default_precision": 1.0 / (shots ** 0.5),
                                    "backend_options": {"noise_model": self.noise_model,
                                                        "method": self.method,
                                                        "seed_simulator": seed, **_threads()},
                                    "run_options": run})

    def sampler(self, shots=4096, seed=0):
        # each shot is one trajectory under "statevector"; exact channel under "density_matrix"
        return SamplerV2(default_shots=shots, seed=seed,
                         options={"backend_options": {"noise_model": self.noise_model,
                                                      "method": self.method,
                                                      "seed_simulator": seed, **_threads()}})

    def param_array(self, values_by_param: dict):
        """Stack parameter values in the compact circuit's parameter order.
        values_by_param maps each original Parameter to a 1-D array of values."""
        import numpy as np
        by_name = {p.name: v for p, v in values_by_param.items()}
        return np.column_stack([np.asarray(by_name[p.name], float) for p in self.parameters])


def make_estimator(mode="sv", shots=4096, seed=0, noise=None):
    if mode == "sv":
        return EstimatorV2(options={"default_precision": 0.0,
                                    "backend_options": {"method": "statevector",
                                                        "seed_simulator": seed, **_threads()}})
    nm = load_noise_model(noise)
    # density-matrix expectation values of the transpiled (ISA) circuit under the device noise
    # model, plus a Gaussian shot-noise proxy of 1/sqrt(shots)
    return EstimatorV2(options={"default_precision": 1.0 / (shots ** 0.5),
                                "backend_options": {"noise_model": nm, "method": "density_matrix",
                                                    "seed_simulator": seed},
                                "run_options": {"seed": seed}})


def make_sampler(mode="sv", shots=4096, seed=0, noise=None):
    # Aer SamplerV2: shots/seed are top-level ctor kwargs (default_shots, seed),
    # NOT options keys - unlike the Runtime SamplerV2.
    if mode == "sv":
        return SamplerV2(seed=seed,
                         options={"backend_options": {"method": "statevector",
                                                      "seed_simulator": seed}})
    nm = load_noise_model(noise)
    return SamplerV2(default_shots=shots, seed=seed,
                     options={"backend_options": {"noise_model": nm, "method": "density_matrix",
                                                  "seed_simulator": seed}})
