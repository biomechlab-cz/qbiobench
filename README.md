# QBioBench

QBioBench is a calibration-aware benchmark of quantum machine learning on peripheral biosignals.
Projected (PQK) and fidelity (FQK) quantum kernels and variational quantum classifiers (VQC) are
compared with classical models on the same low-dimensional features, screened in device-noise
simulation, and executed on IBM quantum hardware. Models are scored on discrimination (ROC-AUC,
balanced accuracy) and calibration (expected calibration error, Brier score), hardware cells also
on cost (executed circuits, QPU time), with cluster-bootstrap intervals and paired comparisons on
identical test samples. The aim is a rigorous comparison, not a demonstration of quantum advantage.

The release contains the code, the raw IBM Quantum job records of the IBM Phoenix hardware
evaluation, the device noise snapshots, every result table, and the capped engineered features of
the two primary tasks. Every hardware prediction is recomputed from the archived job records, so
every released number can be regenerated without a quantum computer and, for the two primary
tasks, without the raw recordings.

Repository: https://github.com/biomechlab-cz/qbiobench

## Tasks

| Task | Dataset | Signal and label | Qubits | Role |
|------|---------|------------------|--------|------|
| T1 | PTB-XL 1.0.3 | 12-lead ECG, five diagnostic superclasses (NORM, MI, STTC, CD, HYP), one-vs-rest | 8 | primary, simulation and hardware |
| T2 | Apnea-ECG 1.0.0 | single-lead ECG, apnea per one-minute segment | 6 | primary, simulation and hardware |
| T3 | WESAD | chest ECG, EDA, respiration, EMG, and temperature, stress versus baseline | 8 | secondary, simulation only |

Features (`utils/features.py`): PTB-XL, 8 leads (I, II, V1 to V6) x 8 SciPy/Welch spectral and
statistical features. Apnea-ECG, 28 HRV and 2 ECG-derived respiration features (NeuroKit2).
WESAD, 36 HRV, EDA (cvxEDA), respiration, EMG, and temperature features. Median imputation,
standardisation, and PCA to one component per qubit are fitted on training data only
(`utils/preprocessing.FeatureReducer`), and the components are scaled to [-pi, pi] for the
encodings. Quantum and classical models see the same PCA features.

## Benchmark design

- Model grid: seven method cells, PQK and FQK with angle or ZZ encoding, and VQC with angle, ZZ,
  or data re-uploading encoding. Every FQK and VQC cell has an entanglement-removed control.
- Classical baselines on the same features: logistic regression, RBF support vector machine,
  XGBoost, and a multilayer perceptron (`experiments/baselines.py`).
- Simulation screen (`experiments/screen.py`): 7 cells x 3 seeds x 3 tasks, with transpiled
  circuits under the noise snapshot of an IBM Heron r3 processor (June 2026) and noiselessly. The
  encoding with the highest mean ROC-AUC within each model family is promoted to hardware for each
  primary task.
- Hardware protocol: 1024 shots, XY4 dynamical decoupling, and two mitigation arms, none and TREX
  (Estimator resilience level 1; the FQK overlap circuits run on the Sampler, which applies
  measurement twirling).
- Statistics (`experiments/stats.py`, one implementation for every number): macro ROC-AUC over the
  one-vs-rest labels of PTB-XL, ECE with ten equal-width bins, Brier score, balanced accuracy at a
  threshold derived from training data only, a cluster bootstrap with 2000 resamples (Apnea-ECG
  recording, PTB-XL patient, WESAD subject), paired differences on identical test samples, and
  Benjamini-Hochberg adjustment within declared families of comparisons. Kernel classifiers are
  ranked by their SVM decision function.

## Hardware evaluation

**IBM Phoenix** (Nighthawk r2, 120 qubits, square lattice), 25 to 26 September 2026, 17 cells
(one cell per configuration and mitigation arm, listed in `HW2_PLAN` in `experiments/config.py`),
exploratory and post hoc, one seed per cell. The kernel cells are FQK/angle with 64 training units
and 16 Nystrom landmarks, PQK/angle with 256 training units, and PQK/ZZ for PTB-XL, and every cell
is scored on the full capped test pool of its task (510 Apnea-ECG minutes from 35 recordings, 512
PTB-XL records from 504 patients). The VQC/angle weights are trained in
simulation with binary cross-entropy on 512 (Apnea-ECG) or 500 (PTB-XL) training units,
noiselessly and against the IBM Phoenix noise snapshot, and the noiselessly trained circuit is
also executed without its two-qubit gates. Every configuration runs unmitigated, and FQK/angle,
PQK/angle, and the noiselessly trained VQC also run with TREX. The run plan is
`data/hardware/phoenix_2026-09/plan.json`. The job records are replayed by
`experiments/hw_replay2.py`, which yields the per-sample predictions, and analysed by
`experiments/analysis_hw.py`.

## Released results

| Path | Content |
|------|---------|
| `data/results/hw/phoenix_2026-09/` | per-cell results, per-sample predictions, and the measured kernels and features of the IBM Phoenix run |
| `data/results/analysis_v2/phoenix_2026-09_pool/` | per-cell metrics with cluster-bootstrap intervals, matched baselines, paired differences with Benjamini-Hochberg adjusted p values, and kernel diagnostics, for IBM Phoenix on the full test pools |
| `data/results/screen/screen_v2/` | simulation screen, every unit, the seed means, and the promotion |
| `data/results/baselines_*.csv` | classical baselines on the capped and full pools |
| `data/hardware/phoenix_2026-09/` | raw IBM Quantum job records (gzip-compressed JSON), job index, run plan, and VQC weights |
| `data/noise/`, `data/calibration/` | device noise snapshots (IBM Heron r3, June 2026, used by the screen, and IBM Phoenix) and the calibration table of the Heron r3 snapshot |

[`data/README.md`](data/README.md) documents every file and column under `data/`.
`./reproduce.sh tables` generates summary tables (`tables/*.csv`) and the data figures
(`figures/*.pdf`) locally from `data/results/`.

## Datasets

The raw datasets are public and are not redistributed here. The repository ships only the capped
engineered features of PTB-XL and Apnea-ECG (`data/features/`, see
[`data/README.md`](data/README.md)). The WESAD features are not included because the WESAD terms
of use do not allow redistribution. Download the datasets from their official pages and please
cite them.

| Task | Dataset | Version | Licence | Official page | Environment variable |
|------|---------|---------|---------|---------------|----------------------|
| T1 | PTB-XL, 12-lead ECG, 5 diagnostic superclasses (one-vs-rest) | 1.0.3 | CC BY 4.0 | https://physionet.org/content/ptb-xl/1.0.3/ (doi:10.13026/kfzx-aw45) | `QBIO_PTBXL` |
| T2 | Apnea-ECG, single-lead ECG, per-minute apnea | 1.0.0 | ODC-By 1.0 | https://physionet.org/content/apnea-ecg/1.0.0/ (doi:10.13026/C23W2R) | `QBIO_APNEA` |
| T3 | WESAD, chest RespiBAN, stress versus baseline (simulation only) | as released 2018 | scientific non-commercial use, credit required | https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html (also https://doi.org/10.24432/C57K5T) | `QBIO_WESAD` |

Set each variable to the folder that holds the layout below (`utils/io.py` resolves the paths):

```
$QBIO_PTBXL/ptbxl_database.csv, scp_statements.csv, records100/...   (PhysioNet layout)
$QBIO_APNEA/data/a01.dat, a01.hea, a01.apn, ..., x35.*                (PhysioNet files in data/)
$QBIO_WESAD/Data/S2/S2.pkl, ..., Data/S17/S17.pkl                    (subject folders inside Data/)
```

The WESAD archive unpacks to `WESAD/S2/S2.pkl`, ..., `WESAD/S17/S17.pkl` without a `Data` level.
Rename the unpacked `WESAD` folder to `Data` (or move the subject folders into a `Data` folder)
and set `QBIO_WESAD` to its parent folder. The Apnea-ECG files can also be fetched with
`wfdb.dl_database("apnea-ecg", "$QBIO_APNEA/data")`. Datasets are only read, never modified.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the pinned stack from
`uv.lock`: Qiskit 2.4.1, qiskit-ibm-runtime 0.47.0, qiskit-aer 0.17.2, qiskit-machine-learning
0.9.0, scikit-learn, XGBoost, NeuroKit2).

```bash
git clone https://github.com/biomechlab-cz/qbiobench.git
cd qbiobench
uv sync
uv run python tests/test_kernel_diag.py     # unit tests for the kernel diagnostics
./reproduce.sh tables                       # summary tables (tables/*.csv) and data figures (figures/*.pdf) from data/results/
./reproduce.sh replay                       # recompute every hardware prediction from the job records
```

On Windows, run `reproduce.sh` from Git Bash or WSL. The `tables` stage needs only the shipped
files. The `replay` stage also needs the PTB-XL metadata table: download `ptbxl_database.csv` and
`scp_statements.csv` from the PTB-XL page on PhysioNet (no waveforms) into a folder and set
`QBIO_PTBXL` to it, otherwise the first PTB-XL cell stops the stage before anything is written.
The replay rewrites `data/results/hw/phoenix_2026-09/`, so compare its `cells.csv` and
`predictions.csv` with the shipped files using `git diff`.

## Reproduction

`reproduce.sh` drives every stage in order. Each stage can also be run on its own, and no stage
spends QPU time.

```bash
./reproduce.sh features    # engineered features, capped and full pools (raw datasets)
./reproduce.sh baselines   # classical baselines on the capped and full pools
./reproduce.sh screen      # 7 cells x 3 seeds x 3 tasks: device-noise, noiseless, one-sided-objective VQC
                           # (resumes from the shipped screen_v2 unit logs, see the table below)
./reproduce.sh replay      # recompute every hardware prediction from data/hardware/ (no IBM account)
./reproduce.sh analysis    # hardware statistics, paired tests, kernel diagnostics, data description
./reproduce.sh tables      # summary tables (tables/*.csv) and data figures (figures/*.pdf), generated locally
./reproduce.sh             # all of the above
```

Parallel simulation uses `JOBS` workers (default 16, `JOBS=8 ./reproduce.sh screen`).
`QBIO_NOISE=<file stem in data/noise/>` selects the device noise snapshot (default
`heron_r3_2026-06-11`, the noise snapshot of an IBM Heron r3 processor, June 2026). The `paper`
stage builds the article from its LaTeX sources, which are not part of this repository, so here it
runs `tables` only.

What runs without the datasets:

| Stage | Needs | Notes |
|-------|-------|-------|
| `tables` | shipped results | runs completely from `data/results/` and `data/hardware/` |
| `replay` | shipped job records and features, plus the PTB-XL metadata table | The replay (`hw_replay2.py`) has no task filter. Its PTB-XL cells read patient ids (the bootstrap clusters) from `ptbxl_database.csv` and `scp_statements.csv`: download these two files from PhysioNet (no waveforms needed) into a folder and set `QBIO_PTBXL` to it. The Apnea-ECG cells need nothing else. |
| `analysis` | shipped features and predictions | The stage ends with `describe_data.py`, which needs the WESAD features and the PTB-XL metadata table (its output `data_description.json` is shipped), so without them run the analysis directly: `uv run python experiments/analysis_hw.py --tag phoenix_2026-09 --scope pool`. |
| `baselines` | features | The stage runs `--task all`, which stops at the missing WESAD features, and the full pools need the datasets. From the shipped features run `uv run python experiments/run_baselines.py --task apnea --subset capped` and `--task ptbxl --subset capped`. |
| `screen` | features | The stage resumes from the shipped unit logs in `data/results/screen/screen_v2/` and only re-ranks them, so it simulates nothing. To recompute the screen, use a fresh tag, for example `uv run python experiments/screen.py --task apnea ptbxl --mode noisy --jobs 16 --tag screen_check` (hours of device-noise simulation), and compare it with the shipped `screen_v2`. WESAD needs the dataset. |
| `features` | raw datasets | rebuilds every feature file |

## Hardware stages (IBM Quantum account required, spends QPU time)

The job records in `data/hardware/` are the reference copy of the IBM Phoenix run. Running the
hardware stages needs an IBM Quantum Platform account with access to the chosen backend.

```bash
export IBM_QUANTUM_TOKEN=...                 # API key; never commit credentials
export QBIO_IBM_INSTANCE=<instance name or CRN>
uv run python scripts/ibm_login.py           # optional: save the account locally
uv run python experiments/hw_run2.py --backend ibm_phoenix --tag <tag> --dry-run      # build circuits, report costs
uv run python experiments/hw_run2.py --backend ibm_phoenix --tag <tag> --budget-min 30
uv run python experiments/hw_archive.py --tag <tag> --jobs <job ids>                   # archive the job records
uv run python experiments/hw_replay2.py --tag <tag>                                    # metrics from the records
```

`scripts/qbio_runtime.get_service()` always pins the instance (`QBIO_IBM_INSTANCE`, or `--instance`
on the command line), because a saved default account in another region cannot see the jobs.
`hw_run2.py` stops before a cell once the run has used `--budget-min` QPU minutes. The VQC weights
are trained in simulation beforehand with `experiments/train_vqc_hw.py`. Job records can only be
downloaded again by the account that ran them.

## Repository map

```
experiments/      models (models.py, encodings.py), device-noise simulation (quantum_backend.NoisyCircuit),
                  simulation screen (screen.py), classical baselines (baselines.py, run_baselines.py),
                  hardware runner (hw_run2.py for IBM Phoenix, train_vqc_hw.py for its VQC weights),
                  job archive and replay (hw_archive.py, hw_replay2.py),
                  analysis (analysis_hw.py, stats.py, kernel_diag.py, pool_subsets.py, describe_data.py),
                  summary tables and data figures (make_tables.py, figures_v2.py),
                  config.py (design grid, caps, seeds)
scripts/          build_features.py, build_noise_model.py, qbio_runtime.py (IBM helpers), ibm_login.py
utils/            dataset loaders (io.py), feature extraction (features.py), train-only preprocessing, splits
tests/            unit tests (kernel-target alignment, closed-form FQK and PQK features)
data/             hardware job records, noise snapshots, calibration, capped features, results (data/README.md)
reproduce.sh      end-to-end driver
CITATION.cff      citation metadata
```

## How to cite

Please cite the article and the software as described in [`CITATION.cff`](CITATION.cff) (GitHub
shows it under "Cite this repository"). Please also cite the datasets you use (PTB-XL,
Apnea-ECG, WESAD) as requested on their official pages.

## Licence

- Code: MIT licence, see [`LICENSE`](LICENSE).
- Data produced by this project (hardware job records, noise snapshots, calibration table, and
  the result files in `data/results/`): Creative Commons Attribution 4.0 International (CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/).
- Shipped features: PTB-XL-derived features under CC BY 4.0, Apnea-ECG-derived features under
  ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/), see [`data/README.md`](data/README.md).

## Contact

Questions and bug reports: please open an issue at https://github.com/biomechlab-cz/qbiobench/issues.
