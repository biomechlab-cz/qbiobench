"""Quantum model classes (the *model* axis of the design grid).

  PQK-QSVM : <X_i>,<Y_i>,<Z_i> per qubit -> 3N classical vector -> standardise -> RBF SVM
             (gamma, C grid-searched by 3-fold CV on the training data).
  FQK-QSVM : fidelity kernel |<phi(x)|phi(x')>|^2 -> SVC(precomputed). Exact statevector kernel
             in `sv` mode, landmark (Nystrom) kernel from shot-sampled overlap circuits in `noisy`.
  VQC      : encoding + ansatz (EfficientSU2 reps=3 circular) or data re-uploading (4 layers),
             parity readout p = (1 + <Z..Z>) / 2, trained by seeded SPSA on binary cross-entropy.

`mode` in {'sv','noisy'}. Noisy mode compiles every circuit for the device snapshot with
`quantum_backend.NoisyCircuit` (transpile to the device ISA + error-aware layout, compact onto
the active qubits, remap the noise model). Logical, untranspiled circuits would pick up errors
only on the gates named by the noise model, so they are never simulated with noise.

VQC objective. `loss="bce"` (default) is binary cross-entropy on p = (1 + <Z..Z>)/2.
`loss="one-sided"` (alias "legacy", the name used in the shipped screen file names) is the
objective of qiskit-machine-learning's NeuralNetworkClassifier(loss="cross_entropy",
one_hot=False) on a one-output parity QNN with 0/1 labels. For a one-dimensional output that
loss evaluates -y * log2(<Z..Z>) per sample, so class-0 samples contribute nothing and the
minimiser is <Z..Z> = +1 for every input: the circuit is not trained to separate the classes.
It is available only for the screen comparison of the two objectives.

All classes are binary with fit / predict_proba; PTB-XL multilabel is one-vs-rest in the caller.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit import QuantumCircuit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

# Angle encodings (Ry, ZZ) require inputs in a bounded angular range, else the
# rotations wrap and destroy structure. Scale features to [-pi, pi], fit on train.
ANGLE_RANGE = (-np.pi, np.pi)

from experiments.encodings import angle_encoding, zz_map, data_reuploading, remove_entanglement
from experiments.quantum_backend import make_estimator, NoisyCircuit
from experiments.config import FQK_LANDMARKS

PQK_GRID = {"gamma": ["scale", 0.01, 0.1, 1.0], "C": [0.1, 1, 10]}


# --- encoding helpers -------------------------------------------------------

def feature_map(name, n):
    """Fixed data-encoding circuit (kernels / VQC input). n data params."""
    if name == "angle":
        return angle_encoding(n)
    if name == "zz":
        return zz_map(n, reps=2, entanglement="linear")
    raise ValueError(f"{name} is not a fixed feature map (reuploading is VQC-only)")


def _pauli(letter, i, n):
    s = ["I"] * n
    s[i] = letter
    return SparsePauliOp(Pauli("".join(reversed(s))))


def _parity_obs(n):
    return SparsePauliOp(Pauli("Z" * n))


def _bind(params, blocks):
    """Parameter-value matrix in `params` order from {Parameter: 1-D array | scalar} blocks."""
    n = max(len(np.atleast_1d(v)) for v in blocks.values())
    by_name = {p.name: v for p, v in blocks.items()}
    return np.column_stack([np.broadcast_to(np.asarray(by_name[p.name], float), (n,)) for p in params])


# =========================== PQK ===========================================
class PQKQSVM:
    def __init__(self, encoding="angle", n_qubits=6, *, mode="sv", shots=4096,
                 seed=0, ablate_ent=False, grid=True):
        self.n = n_qubits
        self.mode = mode; self.shots = shots; self.seed = seed; self.grid = grid
        circ = feature_map(encoding, n_qubits)
        self.circ = remove_entanglement(circ) if ablate_ent else circ
        self.obs = [_pauli(L, i, n_qubits) for i in range(n_qubits) for L in "XYZ"]
        self._nc = None

    def projected_features(self, X):
        X = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        xs = list(self.circ.parameters)
        if self.mode == "sv":
            est = make_estimator("sv", self.shots, self.seed)
            circ, obs = self.circ, self.obs
        else:
            if self._nc is None:
                self._nc = NoisyCircuit(self.circ, seed=0)
            circ = self._nc.circuit
            obs = [self._nc.observable(o) for o in self.obs]
            est = self._nc.estimator(self.shots, self.seed)
            xs = self._nc.parameters
        order = {p.name: k for k, p in enumerate(self.circ.parameters)}
        pv = X[:, [order[p.name] for p in xs]]
        obs_arr = np.empty((len(obs), 1), dtype=object)
        for i, o in enumerate(obs):
            obs_arr[i, 0] = o
        evs = est.run([(circ, obs_arr, pv[np.newaxis, :, :])]).result()[0].data.evs   # (3N, n)
        return np.asarray(evs).T.reshape(len(X), 3 * self.n)

    def fit(self, X, y):
        self.in_scaler = MinMaxScaler(ANGLE_RANGE).fit(np.asarray(X, float)[:, : self.n])
        F = self.projected_features(X)
        self.scaler = StandardScaler().fit(F)
        F = self.scaler.transform(F)
        if self.grid:
            g = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=self.seed),
                             PQK_GRID, cv=3, n_jobs=1)
            self.svm = g.fit(F, y).best_estimator_
        else:
            self.svm = SVC(kernel="rbf", probability=True, random_state=0).fit(F, y)
        return self

    def predict_proba(self, X):
        F = self.scaler.transform(self.projected_features(X))
        return self.svm.predict_proba(F)[:, 1]


# =========================== FQK ===========================================
def _overlap_template(fmap):
    """U(x) U(y)^dagger with measurement; P(0...0) is the fidelity kernel k(x, y)."""
    n = fmap.num_qubits
    pb = ParameterVector("y", n)
    inv = fmap.inverse().assign_parameters({p: pb[i] for i, p in enumerate(fmap.parameters)})
    t = QuantumCircuit(n, n)
    t.compose(fmap, inplace=True); t.compose(inv, inplace=True); t.measure(range(n), range(n))
    return t, list(fmap.parameters), list(pb)


class FQKQSVM:
    def __init__(self, encoding="angle", n_qubits=6, *, mode="sv", shots=4096,
                 seed=0, ablate_ent=False, landmarks=FQK_LANDMARKS):
        self.n = n_qubits; self.mode = mode; self.shots = shots; self.seed = seed
        # Exact FQK is O(n^2) kernel circuits - the scalable alternative is a landmark /
        # Nystrom approximation. Applied only in noisy mode when n_train exceeds `landmarks`.
        self.landmarks = landmarks
        circ = feature_map(encoding, n_qubits)
        self.fmap = remove_entanglement(circ) if ablate_ent else circ
        self._nc = None

    # ---- kernel evaluation ----
    def _sv_states(self, X):
        from qiskit.quantum_info import Statevector
        return np.array([Statevector(self.fmap.assign_parameters(x)).data for x in X])

    def kernel(self, XA, XB):
        if self.mode == "sv":
            SA, SB = self._sv_states(XA), self._sv_states(XB)
            return np.abs(SA.conj() @ SB.T) ** 2
        if self._nc is None:
            t, self._a, self._b = _overlap_template(self.fmap)
            self._nc = NoisyCircuit(t, seed=0)
        ai = np.repeat(np.arange(len(XA)), len(XB)); bi = np.tile(np.arange(len(XB)), len(XA))
        blocks = {**{p: XA[ai, k] for k, p in enumerate(self._a)},
                  **{p: XB[bi, k] for k, p in enumerate(self._b)}}
        pv = _bind(self._nc.parameters, blocks)
        res = self._nc.sampler(self.shots, self.seed).run([(self._nc.circuit, pv)]).result()[0]
        creg = res.data.c
        zero = "0" * self.n
        f = np.array([creg[j].get_counts().get(zero, 0) / self.shots for j in range(len(ai))])
        return f.reshape(len(XA), len(XB))

    def fit(self, X, y):
        self.in_scaler = MinMaxScaler(ANGLE_RANGE).fit(np.asarray(X, float)[:, : self.n])
        self.Xtr = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        self.nystrom = self.mode == "noisy" and self.landmarks and len(self.Xtr) > self.landmarks
        if self.nystrom:
            rng = np.random.default_rng(self.seed)
            self.L = self.Xtr[rng.choice(len(self.Xtr), self.landmarks, replace=False)]
            K_LL = self.kernel(self.L, self.L)
            w, V = np.linalg.eigh((K_LL + K_LL.T) / 2.0)     # symmetric sqrt-pinv
            keep = w > 1e-10
            self.Wh = (V[:, keep] * (1.0 / np.sqrt(w[keep]))) @ V[:, keep].T
            self.Phi_tr = self.kernel(self.Xtr, self.L) @ self.Wh
            G = self.Phi_tr @ self.Phi_tr.T                  # PSD by construction
        else:
            G = self.kernel(self.Xtr, self.Xtr)
        self.G_tr = G
        self.svm = SVC(kernel="precomputed", probability=True, random_state=self.seed)
        self.svm.fit(G, y)
        return self

    def predict_proba(self, X):
        Xte = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        if self.nystrom:
            G = (self.kernel(Xte, self.L) @ self.Wh) @ self.Phi_tr.T
        else:
            G = self.kernel(Xte, self.Xtr)
        return self.svm.predict_proba(G)[:, 1]


# =========================== VQC ===========================================
def _bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _legacy_ce(ev, y):
    """The one-sided objective: qiskit-machine-learning CrossEntropyLoss on a one-output
    parity QNN with 0/1 targets = mean(-y * log2(clip(<Z..Z>, 1e-10)))."""
    return float(np.mean(-y * np.log2(np.clip(ev, 1e-10, None))))


class VQCReuploading:
    """Variational classifier. Name kept for backward compatibility; covers every VQC cell."""

    def __init__(self, encoding="reuploading", n_qubits=6, *, reps=3, max_iter=200,
                 mode="sv", train_mode=None, shots=4096, seed=0, ablate_ent=False,
                 loss="bce", record_every=10):
        self.n = n_qubits; self.mode = mode; self.shots = shots; self.seed = seed
        self.max_iter = max_iter; self.loss = loss; self.record_every = record_every
        # Protocol (fixed before any result): parameters trained noiselessly (sv) and evaluated
        # under the target noise. train_mode="noisy" is the noise-aware control.
        self.train_mode = train_mode or "sv"
        self.encoding = encoding
        self.circ, self.in_p, self.w_p = self._build(encoding, n_qubits, reps, ablate_ent)
        self._nc = None
        self.history = []

    def _build(self, encoding, n, reps, ablate_ent):
        if encoding == "reuploading":
            circ = data_reuploading(n, layers=4)
            xs = [p for p in circ.parameters if p.name.startswith("x")]
            ws = [p for p in circ.parameters if not p.name.startswith("x")]
        else:
            fm = feature_map(encoding, n)
            circ = fm.compose(efficient_su2(n, reps=reps, entanglement="circular"))
            xs = list(fm.parameters)
            ws = [p for p in circ.parameters if p not in set(xs)]
        if ablate_ent:
            circ = remove_entanglement(circ)
        return circ, xs, ws

    @property
    def n_weights(self):
        return len(self.w_p)

    def _ev(self, Xs, w, mode):
        """Parity expectation <Z..Z> for scaled inputs Xs under weights w."""
        blocks = {**{p: Xs[:, k] for k, p in enumerate(self.in_p)},
                  **{p: np.full(len(Xs), w[k]) for k, p in enumerate(self.w_p)}}
        if mode == "sv":
            est = make_estimator("sv", self.shots, self.seed)
            circ, obs, params = self.circ, _parity_obs(self.n), list(self.circ.parameters)
        else:
            if self._nc is None:
                self._nc = NoisyCircuit(self.circ, seed=0)
            circ, obs, params = self._nc.circuit, self._nc.observable(_parity_obs(self.n)), self._nc.parameters
            est = self._nc.estimator(self.shots, self.seed)
        pv = _bind(params, blocks)
        return np.asarray(est.run([(circ, [obs], pv)]).result()[0].data.evs).ravel()

    def _objective(self, ev, y):
        return _legacy_ce(ev, y) if self.loss in ("legacy", "one-sided") else _bce((1.0 + ev) / 2.0, y)

    def fit(self, X, y, *, X_val=None, y_val=None):
        from qiskit_machine_learning.optimizers import SPSA
        from qiskit_machine_learning.utils import algorithm_globals
        from sklearn.metrics import roc_auc_score
        self.in_scaler = MinMaxScaler(ANGLE_RANGE).fit(np.asarray(X, float)[:, : self.n])
        Xs = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        y = np.asarray(y, float)
        Xv = self.in_scaler.transform(np.asarray(X_val, float)[:, : self.n]) if X_val is not None else None
        algorithm_globals.random_seed = self.seed           # SPSA perturbations (seeded)
        w0 = np.random.default_rng(self.seed).random(self.n_weights)   # qml default init U[0,1)
        self.history = []
        it = {"k": 0}

        def f(w):
            return self._objective(self._ev(Xs, w, self.train_mode), y)

        def cb(nfev, w, fx, step, accepted):
            it["k"] += 1
            row = {"iter": it["k"], "nfev": int(nfev), "train_loss": float(fx)}
            if self.record_every and (it["k"] % self.record_every == 0 or it["k"] == 1):
                ev_tr = self._ev(Xs, w, self.train_mode)
                row["train_auc"] = float(roc_auc_score(y, ev_tr)) if len(np.unique(y)) > 1 else np.nan
                row["mean_parity"] = float(np.mean(ev_tr))
                if Xv is not None:
                    ev_v = self._ev(Xv, w, self.train_mode)
                    row["val_loss"] = self._objective(ev_v, np.asarray(y_val, float))
                    row["val_auc"] = float(roc_auc_score(y_val, ev_v)) if len(np.unique(y_val)) > 1 else np.nan
            self.history.append(row)

        opt = SPSA(maxiter=self.max_iter, callback=cb)
        res = opt.minimize(f, w0)
        self.weights = np.asarray(res.x)
        self.fit_result = {"fun": float(res.fun), "nfev": int(res.nfev), "nit": int(res.nit)}
        return self

    def predict_proba(self, X):
        Xs = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        return (1.0 + self._ev(Xs, self.weights, self.mode)) / 2.0

    def predict_ev(self, X, mode=None):
        Xs = self.in_scaler.transform(np.asarray(X, float)[:, : self.n])
        return self._ev(Xs, self.weights, mode or self.mode)


MODEL_CLASSES = {"pqk": PQKQSVM, "fqk": FQKQSVM, "vqc": VQCReuploading}
