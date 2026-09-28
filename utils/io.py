"""Dataset path resolution and raw loaders (verified against on-disk formats 2026-06-09).

Datasets are treated as READ-ONLY. Loaders return raw signals +
labels + grouping keys; feature extraction happens in utils.features, PCA in the
experiment (train-only). Loaders are subset-aware so we never hold all of PTB-XL in RAM.
"""
from __future__ import annotations

import ast
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# Dataset roots. Each can be set explicitly with an environment variable
# (QBIO_PTBXL, QBIO_APNEA, QBIO_WESAD); otherwise the first existing candidate is used.
# Candidates cover the original Linux layout (~/Work/Datasets) and the Windows D: layout.
DATASETS_ROOT = Path(os.path.expanduser("~/Work/Datasets"))
_CANDIDATES = {
    "ptbxl": [DATASETS_ROOT / "Misc" / "PTB-XL", Path(r"D:\Misc Datasets\PTB-XL")],
    "apnea": [DATASETS_ROOT / "Misc" / "Apnea-ECG", Path(r"D:\Misc Datasets\Apnea-ECG")],
    "wesad": [DATASETS_ROOT / "CL" / "WESAD", Path(r"D:\CL Datasets\WESAD")],
}
_ENV = {"ptbxl": "QBIO_PTBXL", "apnea": "QBIO_APNEA", "wesad": "QBIO_WESAD"}
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def dataset_path(key: str) -> Path:
    env = os.environ.get(_ENV[key])
    cands = [Path(env)] if env else _CANDIDATES[key]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"{key} dataset not found (tried {', '.join(map(str, cands))}; "
                            f"set {_ENV[key]})")


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# --- T1 PTB-XL --------------------------------------------------------------

def ptbxl_metadata() -> pd.DataFrame:
    """Return the PTB-XL table with a multilabel superclass matrix and strat_fold.
    Columns added: one 0/1 column per SUPERCLASSES entry, plus n_super (count)."""
    root = dataset_path("ptbxl")
    # PhysioNet layout has the CSVs at the root; the original local copy had them in labels/
    df = pd.read_csv(_first_existing(root / "ptbxl_database.csv", root / "labels" / "ptbxl_database.csv"),
                     index_col="ecg_id")
    scp = pd.read_csv(_first_existing(root / "scp_statements.csv", root / "labels" / "scp_statements.csv"),
                      index_col=0)
    diag = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()

    def to_super(scp_codes_str):
        codes = ast.literal_eval(scp_codes_str)
        return {diag[c] for c in codes if c in diag}

    supers = df.scp_codes.apply(to_super)
    for sc in SUPERCLASSES:
        df[sc] = supers.apply(lambda s, sc=sc: int(sc in s)).astype("int8")
    df["n_super"] = df[SUPERCLASSES].sum(axis=1).astype("int16")
    return df


def ptbxl_signal_path(filename_lr: str) -> str:
    """Absolute WFDB record path (no extension) for a filename_lr value."""
    root = dataset_path("ptbxl")
    base = root / "data" if (root / "data" / "records100").exists() else root
    return str(base / filename_lr)


def iter_ptbxl_signals(filenames):
    """Yield (ecg_id_index, signal[1000,12]) for each filename_lr, loaded lazily."""
    for fn in filenames:
        sig, _ = wfdb.rdsamp(ptbxl_signal_path(fn))
        yield fn, sig.astype(np.float32)


# --- T2 Apnea-ECG -----------------------------------------------------------

APNEA_TRAIN = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] \
    + [f"c{i:02d}" for i in range(1, 11)]                 # 35 released records
APNEA_TEST = [f"x{i:02d}" for i in range(1, 36)]          # 35 withheld test records


def apnea_records():
    """Base records present on disk (drops the r/er respiration variants)."""
    root = dataset_path("apnea") / "data"
    recs = sorted(Path(f).stem for f in glob.glob(str(root / "*.dat")))
    return [r for r in recs if r[-1].isdigit()]


def load_apnea_record(rec: str):
    """Return (ecg[N], fs, minute_labels[M] in {0,1}, recording_id).
    Label per minute: 'A' (apnea)->1, 'N' (normal)->0 from the .apn annotation."""
    root = dataset_path("apnea") / "data"
    sig, meta = wfdb.rdsamp(str(root / rec))
    ann = wfdb.rdann(str(root / rec), "apn")
    labels = np.array([1 if s == "A" else 0 for s in ann.symbol], dtype="int8")
    return sig[:, 0].astype(np.float32), int(meta["fs"]), labels, rec


# --- T3 WESAD ---------------------------------------------------------------

WESAD_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                  "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
WESAD_CHEST_FS = 700


def load_wesad_subject(subject: str):
    """Return (signals: dict[channel]->1d array, label[N], fs). Chest RespiBAN only."""
    base = dataset_path("wesad")
    root = _first_existing(base / "data" / subject / f"{subject}.pkl",
                           base / "Data" / subject / f"{subject}.pkl")
    with open(root, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = {k: np.asarray(v).squeeze().astype(np.float32)
             for k, v in d["signal"]["chest"].items()}
    label = np.asarray(d["label"]).astype("int8")
    return chest, label, WESAD_CHEST_FS
