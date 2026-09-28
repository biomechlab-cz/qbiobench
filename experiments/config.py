"""Experiment configuration (design grid, splits, caps, seeds, hardware budget caps).

Single source of truth for caps, dims, grid, seeds. Changing anything here changes
the experiment design, not an implementation detail.
"""
from __future__ import annotations

# --- Design grid (the axes) -------------------------------------------------
ENCODINGS = ["angle", "zz", "reuploading"]        # headline (amplitude = appendix)
MODELS = ["pqk", "fqk", "vqc"]
MITIGATIONS = ["none", "trex", "zne_trex_twirl"]  # HARDWARE ONLY (Phase 3); NOT in the sim screen
SEEDS = [0, 1, 2]
REPLICATION_SEEDS = [0, 1, 2, 3, 4]               # strongest cell only

# 7 valid (model, encoding) method cells. reuploading is VQC-only: it is an
# inherently trainable circuit, not a fixed kernel feature map (decision 2026-06-09).
METHOD_CELLS = [
    ("pqk", "angle"), ("pqk", "zz"),
    ("fqk", "angle"), ("fqk", "zz"),
    ("vqc", "angle"), ("vqc", "zz"), ("vqc", "reuploading"),
]

# --- Per-task spec ----------------------------------------------------------
# qubits/pca_dim are fixed per task; caps bound IBM cost (preprocessing caps + budget gates).
TASKS = {
    "ptbxl": {
        "qubits": 8, "pca_dim": 8, "feat_dim": 64,
        "type": "multilabel", "n_classes": 5,
        "classes": ["NORM", "MI", "STTC", "CD", "HYP"],
        "fs": 100,
        "caps": {"train_per_label": 256, "val_per_label": 128, "test_total": 512},
    },
    "apnea": {
        "qubits": 6, "pca_dim": 6, "feat_dim": 30,
        "type": "binary", "n_classes": 2,
        "fs": 100,
        "caps": {"train_total": 512, "val_total": 256, "test_total": 512},
    },
    "wesad": {
        "qubits": 8, "pca_dim": 8, "feat_dim": 36,
        "type": "binary", "n_classes": 2,
        "fs": 700,
        "caps": {"train_windows_per_subject": 40},
        "win_s": 60.0, "overlap": 0.25,
        "keep_labels": {1: 0, 2: 1},  # baseline->0, stress->1
    },
}

SHOTS = 4096           # default per-circuit shot budget (hardware cap)
BOOTSTRAP_RESAMPLES = 1000
FDR_Q = 0.05

# --- Phase-2 noisy-screen tractability (fixed before any result) ------------
# Noisy Aer is ~40x sim cost; the screen is a *ranking* exercise, not the final
# powered comparison (that is replication/hardware). Per-task caps keep the screen
# to hours; FQK uses a landmark/Nystrom kernel in noisy mode. PTB-XL is smaller
# because multilabel runs 5 OvR fits per cell.
SCREEN_CAPS = {
    "apnea": {"train": 128, "test": 128},
    "wesad": {"train": 128, "test": 128},
    "ptbxl": {"train": 64,  "test": 128},   # multilabel: 5 OvR fits per cell
}
FQK_LANDMARKS = 32     # Nystrom landmark count for noisy FQK (sv stays exact)

# --- Fixed hardware protocol per primary task -------------------------------------------------
# Training sizes, Nystrom landmarks, the sizes of the fixed seeded test subsets (n_test, drawn by
# screen.load_task_split with seed 0), and the mitigation arms, sized to a <=2,500-circuit /
# <=12-min budget gate per cell. Mitigation: kernels {none, TREX} only (ZNE over-runs the circuit
# cap on kernels). DD on for all. Quantum kernels (FQK) and PQK features are LABEL-INDEPENDENT:
# for PTB-XL multilabel the kernel/feature circuits are computed ONCE and reused across the 5
# one-vs-rest SVMs (only the classical fit repeats). The kernel sizes of HW2_PLAN below are the
# same. The VQC rows fix the VQC subset sizes only (512 caps the 510-minute Apnea-ECG pool); the
# IBM Phoenix VQC cells use the encoding promoted by screen_v2 (angle). describe_data.py and
# analysis_hw.py read the sizes from here.
HW_PLAN = {
    "apnea": [   # T2 binary, 6q
        {"model": "fqk", "encoding": "angle", "L": 16, "n_train": 64, "n_test": 80, "mitig": ["none", "trex"]},
        {"model": "pqk", "encoding": "angle", "n_train": 256, "n_test": 416, "mitig": ["none", "trex"]},
        {"model": "vqc", "encoding": "reuploading", "n_test": 512, "mitig": ["none", "trex"]},
    ],
    "ptbxl": [   # T1 multilabel 5-class, 8q (kernels label-independent; vqc x5)
        {"model": "fqk", "encoding": "angle", "L": 16, "n_train": 64, "n_test": 80, "mitig": ["none", "trex"]},
        {"model": "pqk", "encoding": "angle", "n_train": 256, "n_test": 416, "mitig": ["none", "trex"]},
        {"model": "vqc", "encoding": "angle", "n_test": 500, "mitig": ["none", "trex"]},
    ],
}
# --- IBM Phoenix (2026-09): exploratory post hoc evaluation on ibm_phoenix (Nighthawk r2) ------
# Kernel protocol of HW_PLAN; the test kernel/features are measured on the FULL capped test pool
# (the fixed test subsets are inside it). VQC = binary cross-entropy objective, weights trained in
# simulation noiselessly ('sv') and noise-aware ('noisy' control, training in device-noise
# simulation), both evaluated on hardware. Listed in execution priority (budget guard stops early).
def _hw2_plan():
    k = {"apnea": {"fqk": dict(L=16, n_train=64, n_test=80), "pqk": dict(n_train=256, n_test=416)},
         "ptbxl": {"fqk": dict(L=16, n_train=64, n_test=80), "pqk": dict(n_train=256, n_test=416)}}
    vqc_enc = {"apnea": "angle", "ptbxl": "angle"}                # screen promotion (screen_v2)
    vqc_n = {"apnea": 512, "ptbxl": 500}
    cells = []
    for mit in ("none",):
        for t in ("apnea", "ptbxl"):
            for m in ("fqk", "pqk"):
                cells.append({"id": f"{t}_{m}_{mit}", "task": t, "model": m, "encoding": "angle",
                              "mitig": mit, **k[t][m]})
    # the screen_v2 promotion for the PTB-XL PQK family is ZZ (0.518 vs angle 0.499, a near tie
    # at chance); PQK/angle is kept above as the kernel-protocol cell of HW_PLAN
    cells.append({"id": "ptbxl_pqk-zz_none", "task": "ptbxl", "model": "pqk", "encoding": "zz",
                  "mitig": "none", **k["ptbxl"]["pqk"]})
    for t in ("apnea", "ptbxl"):
        for tr in ("sv", "noisy"):
            cells.append({"id": f"{t}_vqc-{tr}_none", "task": t, "model": "vqc", "encoding": vqc_enc[t],
                          "mitig": "none", "vqc_train": tr, "n_train": vqc_n[t], "n_test": vqc_n[t]})
    for t in ("apnea", "ptbxl"):
        for m in ("fqk", "pqk"):
            cells.append({"id": f"{t}_{m}_trex", "task": t, "model": m, "encoding": "angle",
                          "mitig": "trex", **k[t][m]})
    for t in ("apnea", "ptbxl"):
        cells.append({"id": f"{t}_vqc-sv_trex", "task": t, "model": "vqc", "encoding": vqc_enc[t],
                      "mitig": "trex", "vqc_train": "sv", "n_train": vqc_n[t], "n_test": vqc_n[t]})
    # entanglement-removed control for the hardware VQC cells
    for t in ("apnea", "ptbxl"):
        cells.append({"id": f"{t}_vqc-sv-noent_none", "task": t, "model": "vqc", "encoding": vqc_enc[t],
                      "mitig": "none", "vqc_train": "sv", "ablate": True, "n_train": vqc_n[t], "n_test": vqc_n[t]})
    return cells


HW2_PLAN = _hw2_plan()
RESILIENCE = {"none": 0, "trex": 1}   # Runtime V2 resilience_level; DD on for both
# Hardware shots: 1024 per circuit. The QPU cost is sampling-dominated, and 1024 shots keep every
# cell under the 12-min cap at a ~2-3% statistical error, fine for these AUC comparisons.
HW_SHOTS = 1024
