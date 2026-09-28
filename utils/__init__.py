"""QBioBench shared preprocessing library.

Stable, dataset-agnostic building blocks reused across all tasks (T1 PTB-XL,
T2 Apnea-ECG, T3 WESAD). The feature sets are defined in features.py and the
split protocols in splits.py; dataset setup is described in README.md.

Submodules:
    io           dataset path resolution + raw loaders
    preprocessing  filtering, windowing, train-only PCA
    features     HRV / EDA / EMG / RSP feature extraction
    splits       split protocols (stratified k-fold, per-recording, LOSO)
"""

DATASETS_ROOT = "~/Work/Datasets"
