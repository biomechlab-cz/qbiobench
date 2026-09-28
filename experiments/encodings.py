"""Quantum feature encodings (the *encoding* axis of the design grid).

Headline: angle, ZZ-feature-map, data-reuploading. Appendix-only: amplitude.
Also provides `remove_entanglement` for the entanglement-removed control:
every 2-qubit gate → identity.

Qiskit 2.4.1: prefer the `zz_feature_map` / `efficient_su2` *functions*.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map


def angle_encoding(n_qubits: int, *, cz_ring: bool = True) -> QuantumCircuit:
    """R_y(x_i) per qubit, optional CZ entangling ring."""
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    if cz_ring and n_qubits > 1:
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)
    return qc


def zz_map(n_qubits: int, *, reps: int = 2, entanglement: str = "linear") -> QuantumCircuit:
    """Havlicek ZZ-feature-map. Report depth + 2Q count downstream."""
    return zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)


def data_reuploading(n_qubits: int, *, layers: int = 4) -> QuantumCircuit:
    """Trainable data re-uploading: per layer R_y(w_l ⊙ x + b_l) + CNOT ring
    (Perez-Salinas 2020). Params: data `x` (n_qubits) + trainable `w`,`b`
    (layers·n_qubits each). Returns the circuit; the VQC binds `w`,`b`."""
    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", layers * n_qubits)
    b = ParameterVector("b", layers * n_qubits)
    qc = QuantumCircuit(n_qubits)
    k = 0
    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(w[k] * x[q] + b[k], q)
            k += 1
        if n_qubits > 1:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)
    return qc


def amplitude_encoding(n_qubits: int):
    """APPENDIX-ONLY ablation (cyclic-padding artifact). Excluded from headline
    tables. Never implemented, so no amplitude result exists."""
    raise NotImplementedError("Appendix ablation only: amplitude encoding")


def remove_entanglement(qc: QuantumCircuit) -> QuantumCircuit:
    """Entanglement-removed control: drop every 2-qubit gate (cz/cx/...) → identity,
    keeping single-qubit structure. Returns a new circuit."""
    out = qc.copy_empty_like()          # preserves exact qubits/registers/params
    for instr in qc.data:
        if instr.operation.num_qubits >= 2:
            continue
        out.append(instr.operation, instr.qubits, instr.clbits)
    return out
