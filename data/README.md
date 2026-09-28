# Data

This folder holds everything needed to recompute the reported results without a quantum computer
and, for PTB-XL and Apnea-ECG, without the raw recordings: the raw IBM Quantum job records, the
device noise snapshots, the capped engineered features, and every result table.

| Path | Content | Produced by | Licence |
|------|---------|-------------|---------|
| `hardware/phoenix_2026-09/` | raw IBM Quantum job records, job index, run plan, trained VQC weights | `experiments/hw_archive.py`, `hw_run2.py`, `train_vqc_hw.py` | CC BY 4.0 |
| `noise/` | device noise snapshots for simulation | `scripts/build_noise_model.py` | CC BY 4.0 |
| `calibration/` | calibration table of the IBM Heron r3 processor of the screen's noise snapshot (June 2026) | IBM Quantum Platform export | CC BY 4.0 |
| `features/` | capped engineered feature pools, PTB-XL and Apnea-ECG | `scripts/build_features.py` | source licences, see below |
| `results/` | hardware predictions, statistics, screen, baselines | `experiments/*.py` (see `reproduce.sh`) | CC BY 4.0 |

Run: `phoenix_2026-09` is the exploratory post hoc run on IBM Phoenix (Nighthawk r2, 25 to 26
September 2026, 17 cells, full test pools). Its result files are described under
[Hardware result files](#hardware-result-files).

## Licences and provenance

- Data produced by this project (job records as archived, noise snapshots, calibration table,
  and the result files in `results/`) are released under the Creative Commons
  Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The job
  records, device properties, and calibration table were retrieved from the IBM Quantum Platform
  with the authors' account.
- `features/ptbxl_capped_*.npz` are derived from PTB-XL v1.0.3 (Wagner et al., PhysioNet,
  https://doi.org/10.13026/kfzx-aw45), which is licensed under CC BY 4.0. Changes: spectral and
  statistical features were computed from the 100 Hz waveforms of a capped subset of records by
  `scripts/build_features.py`. The derived features are distributed under CC BY 4.0.
- `features/apnea_capped_*.npz` contain information from the Apnea-ECG Database (Penzel et al.,
  PhysioNet, https://doi.org/10.13026/C23W2R), which is made available under the Open Data Commons
  Attribution License v1.0 (ODC-By 1.0, https://opendatacommons.org/licenses/by/1-0/). The derived
  features are distributed under the same licence.
- WESAD features are not included. The WESAD terms of use
  (https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) permit use for scientific,
  non-commercial purposes with credit to the owners and grant no right to redistribute the data or
  data derived from it. Obtain WESAD from its owners, place the subject folders `S2` to `S17`
  in a folder named `Data` (the released archive unpacks to `WESAD/S2/S2.pkl` and so on, so rename
  the unpacked `WESAD` folder to `Data`), set `QBIO_WESAD` to the parent of `Data`, and run
  `uv run python scripts/build_features.py --task wesad --subset capped` to create
  `features/wesad_capped.npz`. The WESAD results (screen and baselines) are included.

Please cite the source datasets when you use the features (see the main README).

## `hardware/phoenix_2026-09/`

| File | Format |
|------|--------|
| `<job_id>.json.gz` | One IBM Quantum Runtime job, gzip-compressed JSON written with `qiskit_ibm_runtime.utils.RuntimeEncoder` (QPY circuits, compressed NumPy arrays). Keys: `job_id`, `session_id`, `backend`, `primitive` (`sampler` or `estimator`), `status`, `creation_date`, `metrics` (provider timestamps and usage, including `quantum_seconds`), `inputs` (`pubs` with the transpiled ISA circuits, the exact executed parameter values and, for Estimator jobs, the observables; `options`; `resilience_level`; `version`), `result` (the `PrimitiveResult`: counts for Sampler jobs, expectation values and standard errors for Estimator jobs), `backend_properties` (device calibration at execution time). |
| `index.json` | List of the archived jobs: `job_id`, `session_id`, `backend`, `primitive`, `status`, `created`, `quantum_seconds`, `file` (the uncompressed name; the loader resolves the `.gz` file). |
| `plan.json` | Run plan written by `hw_run2.py`: `backend`, `tag`, `shots`, `probes` (cost probe jobs), `bell` (Bell probe value, job, and session of each session), and `cells`, keyed `<task>_<model>_<mitigation>`, each with `task`, `model`, `encoding`, `mitig`, `n_train`, `n_test`, `L` and `landmark_idx` (FQK Nystrom landmarks), `sub_idx` (positions of the fixed test subset of the cell inside the full test pool, see `in_sub` below), `jobs` (list of `role`, `job`, `n_pubs`), `circuits`, `two_qubit`, `depth`, `quantum_seconds`, `wall_s`, `session`, `status`, and for VQC cells `vqc_train` (`sv` noiseless or `noisy` noise-aware training) and `ablate` (two-qubit gates removed). Job roles: FQK `LL`, `trL`, `teL` (landmark, training, and test blocks of the Nystrom kernel), PQK `tr`, `te`, VQC `label<k>` (one job per one-vs-rest label). |
| `vqc_<task>_angle_<train>[_noent]_label<k>.npz` | VQC weights trained in simulation before the session (`experiments/train_vqc_hw.py`): `weights` (float64), `history` (JSON string, SPSA training history), `fit` (JSON string, optimiser summary). `<train>` is `sv` or `noisy`, `_noent` marks the control without two-qubit gates. |

Load a record with `experiments.hw_archive.load_archived(path)`, which accepts `.json` and
`.json.gz` and returns Qiskit objects. Only the account that ran the jobs can download them again
from IBM (`hw_archive.py`), so these records are the reference copy.

## `noise/`

`heron_r3_2026-06-11.pkl` (noise snapshot of an IBM Heron r3 processor, June 2026, device
properties of 11 June 2026) and `ibm_phoenix_2026-09-25.pkl` (live IBM Phoenix properties of
25 September 2026). Each is a pickled dict with `noise_model` (Qiskit Aer noise model as a dict),
`target` (Qiskit `Target` of the device), `properties`, `configuration`, `basis_gates`, and
`source`, with the device metadata kept as retrieved from the IBM Quantum Platform. They are used
by `experiments/quantum_backend.py` (select with `QBIO_NOISE=<file stem>`, default
`heron_r3_2026-06-11`). The simulation screen uses the Heron r3 snapshot, and the noise-aware VQC
training of the IBM Phoenix run uses the IBM Phoenix snapshot. Unpickling needs the pinned Qiskit
versions (`uv.lock`), and a pickle can execute code, so load only copies from this repository.

## `calibration/`

`heron_r3_calibrations_2026-06-11.csv`: the calibration table of 11 June 2026 of the IBM Heron r3
processor of the noise snapshot, as exported from the IBM Quantum Platform, one row per qubit (156)
with T1, T2, readout errors, single-qubit and CZ/RZZ gate errors and lengths, and an `Operational`
flag. It documents the device state behind the screen's noise snapshot. The code reads the device
properties from the snapshot instead.

## `features/`

Raw engineered features before any imputation, scaling, or PCA (median imputation,
standardisation, PCA, and scaling to [-pi, pi] are fitted on training data only inside each
experiment by `utils/preprocessing.FeatureReducer`). Rows with every feature missing are dropped.

| File | Arrays | Content |
|------|--------|---------|
| `apnea_capped_train.npz` | `X` float32 (764, 30), `y` int64, `groups` str, `feat_names` str (30) | Apnea-ECG released training recordings a01 to a20, b01 to b05, c01 to c10 (35). Class-balanced sample of at most 768 one-minute segments. `y` = 1 for an apnea minute. `groups` = recording id. Features: 28 HRV and 2 ECG-derived respiration features (NeuroKit2). |
| `apnea_capped_test.npz` | `X` float32 (510, 30), `y`, `groups`, `feat_names` | Apnea-ECG test recordings x01 to x35 (35), class-balanced sample of at most 512 minutes. This is the full test pool evaluated on IBM Phoenix. |
| `ptbxl_capped_dev.npz` | `X` float32 (1841, 64), `y` int8 (1841, 5), `folds` int64, `ecg_id` int64, `X_spectrum` float32 (1841, 396), `feat_names` str (64) | PTB-XL stratified folds 1 to 8, capped per superclass. `y` columns NORM, MI, STTC, CD, HYP (one-vs-rest labels). Features: 8 leads (I, II, V1 to V6) x 8 SciPy/Welch spectral and statistical features. `X_spectrum` is the lead-II rFFT magnitude spectrum (0.5 to 40 Hz) of an alternative VQC input path, which the reported models do not use. |
| `ptbxl_capped_test.npz` | same arrays, 512 rows | PTB-XL folds 9 and 10, random sample of 512 records. This is the full test pool evaluated on IBM Phoenix. |

The cluster bootstrap resamples Apnea-ECG recordings (`groups`) and PTB-XL patients. PTB-XL
patient ids are not stored in the feature files. `experiments/screen.load_task_split(...,
return_groups=True)` reads them through `ecg_id` from the PTB-XL metadata table
(`ptbxl_database.csv`, together with `scp_statements.csv`), so the hardware replay of the PTB-XL
cells needs these two CSV files (a few MB from PhysioNet, no waveforms) in the folder named by
`QBIO_PTBXL`. The analysis stage reads the patient ids from
`results/hw/phoenix_2026-09/predictions.csv`.

## `results/`

### Hardware result files

The hardware results are CSV files in two folders. `hw/phoenix_2026-09/` holds the cells
(`cells.csv`) and the per-sample predictions (`predictions.csv`) of the IBM Phoenix run, and
`analysis_v2/phoenix_2026-09_pool/` holds the statistics computed from them on the full capped test
pools (510 Apnea-ECG minutes from 35 recordings, 512 PTB-XL records from 504 patients), written by
`experiments/analysis_hw.py --tag phoenix_2026-09 --scope pool` with every metric and interval
from `experiments/stats.py`.

Conventions shared by all files:

- Keys. `model` is `fqk` (fidelity kernel, angle encoding), `pqk` (projected kernel, angle
  encoding), `pqk-zz` (projected kernel, ZZ encoding), `vqc-sv` (variational classifier trained
  noiselessly with binary cross-entropy), `vqc-noisy` (the same objective trained against the
  device noise snapshot), or `vqc-sv-noent` (the `vqc-sv` weights executed without two-qubit
  gates). `mitig` is `none` or `trex` (TREX for the PQK and VQC circuits, measurement twirling for
  the FQK circuits, which run on the Sampler). `task` is `apnea` (Apnea-ECG, binary) or `ptbxl`
  (PTB-XL, five superclasses one-vs-rest).
- ROC-AUC (`roc_auc`) ranks the test units by the `score` column of `predictions.csv`, the SVM
  decision function for the kernel classifiers and the RBF support vector machine baseline, the
  predicted probability for the other baselines, and p = (1 + <Z...Z>)/2 for the variational
  classifiers. `roc_auc_platt` ranks by the Platt-scaled probability instead. `pr_auc` is the
  average precision of the same ranking scores.
- Intervals. `auc_lo_cluster` and `auc_hi_cluster`, and `lo` and `hi` in `paired.csv`, are 95%
  percentile intervals of a cluster bootstrap with 2000 resamples of whole clusters (Apnea-ECG
  recordings, PTB-XL patients). The `_iid` columns resample single test units instead and are
  given for reference.
- Balanced accuracy (`bal_acc`) and the F1 score of the positive class (`f1`) use the frozen
  threshold in `threshold`. For the kernel classifiers and the baselines it is the threshold on the
  ranking score that maximises balanced accuracy on out-of-fold training predictions (candidates
  are the 5% to 95% quantiles of those scores in steps of 5%), for the variational classifiers it
  is 0.5 on p.
- Calibration. `ece` is the expected calibration error with ten equal-width probability bins and
  `brier` the Brier score, both from the probabilities (`proba`, Platt-scaled for the support
  vector machines).
- PTB-XL. Every metric is the macro average over the five one-vs-rest superclasses (so `f1` is
  the macro-F1), and `threshold` lists one value per superclass, `;`-separated in the order NORM,
  MI, STTC, CD, HYP.
- Sizes and cost. `n_train` training units, `n_test` test units, `n_clusters` clusters
  (recordings or patients) among the test units, `circuits` executed parameter sets, `qpu_min`
  provider-reported QPU minutes of the whole cell, `bell` the Bell probe value P(00) + P(11) of
  the session that ran the cell.

### `hw/phoenix_2026-09/` (written by `hw_replay2.py`)

Every cell scores the full test pool of its task. Each cell also carries a fixed test subset of
that pool, the seeded `n_test` draw of `experiments/screen.load_task_split` (seed 0, positions in
`plan.json` `sub_idx`): 80 units for FQK, 416 for PQK, 500 PTB-XL records for the VQC, and the whole
510-minute pool for the Apnea-ECG VQC. The subset columns below refer to it.

`cells.csv`, one row per hardware cell (task, model, and mitigation arm): `tag`, `task`, `model`,
`encoding`, `mitig`, `roc_auc` (decision scores, on the fixed test subset of the cell),
`roc_auc_pool` (decision scores, on the full test pool), `roc_auc_logged` (the executed
probabilities `proba`, on the fixed test subset), `n_train`, `n_test` (units of the fixed test
subset), `n_pool` (units of the full test pool), `circuits` (executed parameter sets), `shots`,
`quantum_seconds`, `qpu_min`, `session`, `jobs` (job ids, `;`-separated), `two_qubit` and `depth`
(transpiled template circuit of the cell as stored in its job records, for the VQC the circuit of
the last label), `align_max_abs_diff` (largest difference between the executed inputs and a
rebuild of the features from the raw datasets), `backend`, `per_class_auc` (PTB-XL, on the fixed
test subset, `;`-separated in the order NORM, MI, STTC, CD, HYP), `bell` (Bell probe value of the
session).

`predictions.csv`, one row per sample, label, and cell: `split` (`test`, or `train_oof` for
out-of-fold training predictions), `idx` (row in the test pool or training sample), `in_sub`
(the test sample belongs to the fixed test subset of the cell, true for every `train_oof` row),
`group` (recording or patient id, the bootstrap cluster), `label` (-1 for the binary apnea task, 0 to 4 for the PTB-XL superclass),
`y_true`, `proba` (probability as executed: Platt-scaled SVM output, or p = (1 + <Z...Z>)/2 for
the VQC), `score` (ranking score: SVM decision function, or p for the VQC), `tag`, `task`,
`model`, `encoding`, `mitig`.

`matrices/<task>_<model>_<mitig>.npz`: the measured quantities per cell. FQK: `K_LL`, `K_trL`,
`K_teL` (Nystrom blocks), `landmarks`, `Xtr_s`, `Xte_s` (executed inputs in radians), `G_tr`,
`G_te` (Nystrom Gram matrices). PQK: `F_tr`, `F_te` (measured <X_i>, <Y_i>, <Z_i> per qubit),
`Xtr_s`, `Xte_s`. VQC: `evs` (parity expectation values per label), `labels`, `Xte_s`. Every file
also holds `in_sub`.

### `analysis_v2/phoenix_2026-09_pool/` (written by `analysis_hw.py`) and `analysis_v2/data_description.json` (written by `describe_data.py`)

Every file of `phoenix_2026-09_pool/` scores the full capped test pools (see
[Hardware result files](#hardware-result-files)).

| File | One row per | Columns |
|------|-------------|---------|
| `cells_metrics.csv` | hardware cell (task, model, mitigation arm) | `task`, `model`, `encoding`, `mitig`, `source` (`hardware`), `backend`, `roc_auc`, `auc_lo_cluster`, `auc_hi_cluster`, `auc_lo_iid`, `auc_hi_iid`, `roc_auc_platt`, `pr_auc`, `ece`, `brier`, `bal_acc`, `f1`, `threshold`, `n_test`, `n_clusters`, `n_train`, `circuits`, `qpu_min`, `bell` |
| `baselines.csv` | hardware cell and classical baseline | Classical models trained on the same training units as the cell and tested on the exact inputs the device executed (recovered from the job records): `task`, `cell` (`<model>/<mitig>`), `baseline` (`logreg` logistic regression, `rbf_svm` RBF support vector machine, `xgboost`, `mlp` multilayer perceptron, `experiments/baselines.py`), then the columns of `cells_metrics.csv` from `roc_auc` to `n_clusters`, each baseline at its own training-derived threshold |
| `paired.csv` | paired comparison on identical test samples | `family` (see below), `task`, `a`, `b` (the compared models), `diff` (a minus b), `lo`, `hi` (cluster bootstrap), `p_gt0` (fraction of resamples with a difference above zero), `p_boot` (two-sided bootstrap p value, twice the smaller tail mass at zero), `n_boot` (resamples with both classes present), `lo_iid`, `hi_iid` (ROC-AUC baseline families only), `primary`, `auc_tuned`, `auc_b`, `extra_qpu_min`, `p_bh` |
| `kernels.csv` | task and training kernel variant | `task`, `model` (`FQK/angle` or `PQK/angle`), `kernel` (exact closed form, Nystrom with 16 landmarks, RBF on the standardised projected features with its bandwidth `gamma` from the scale rule 1/(d Var X), or linear Gram of the raw projected features), `source` (`noiseless`, from the executed training inputs, or `hardware`, from the measured kernel entries or features), `kta_centered` (centered kernel-target alignment of Cortes et al., macro average over the PTB-XL superclasses), `eff_rank` (exponential of the spectral entropy), `num_rank` (eigenvalues above 1e-10 times the largest), `k95` (eigenvalues holding 95% of the trace), `lambda_min`, `condition`, `n` (training units), `pqk_max_abs_Y` (largest absolute noiseless <Y_i>, zero in closed form). Training kernels do not depend on the test set |
| `kernel_agreement.csv` | task and mitigation arm of FQK/angle | Measured kernel entries against the closed form prod cos^2((x_i - x'_i)/2) on the same executed inputs, per Nystrom block, `LL` (landmarks), `trL` (training units against landmarks), and `teL` (test units against landmarks): `<block>_mae` (mean absolute error), `<block>_corr` (Pearson correlation), `<block>_slope` (least-squares slope of measured on closed form) |
| `pqk_feature_agreement.csv` | task and mitigation arm of PQK/angle | Measured projected features against the closed form (`experiments/kernel_diag.pqk_angle_closed_form`) on the same executed inputs, for the training (`tr`) and test (`te`) units: `<side>X_corr`, `<side>X_mae`, `<side>Z_corr`, `<side>Z_mae` (Pearson correlation and mean absolute error of the <X_i> and <Z_i> columns), `<side>Y_sd` (standard deviation of the measured <Y_i>, whose closed form is zero) |
| `sensitivity.csv` | unmitigated cell and test-set size `n` | ROC-AUC of the fixed trained model on random test subsets of size `n` drawn without replacement from the full test pool: `task`, `model`, `source`, `n`, `mean`, `sd`, `p2.5`, `p97.5`, `reps` (number of draws, 1000, or one at the full size) |
| `composition.csv` | unmitigated cell | `task`, `cell` (model key), `n_test`, `n_clusters`, `per_cluster_min`, `per_cluster_median`, `per_cluster_max` (test units per recording or patient), `pos_frac` (fraction of positive labels, averaged over the superclasses for PTB-XL) |
| `../data_description.json` | (one JSON object) | Pool sizes, missingness, per-subject WESAD counts, and the composition of every screen, hardware, and pool subset |

Families of `paired.csv` (`a` minus `b`, on the full test pool):

| `family` | Comparison |
|----------|------------|
| `qml_vs_baseline`, `qml_vs_baseline_trex` | ROC-AUC of the hardware cell `a` = `<model>/<mitig>` (unmitigated or mitigated arm) minus each matched baseline `b` (`logreg`, `rbf_svm`, `xgboost`, `mlp`) |
| `qml_vs_baseline_ba`, `qml_vs_baseline_ba_trex` | balanced accuracy of the same comparison, each model at its own frozen threshold |
| `trex_minus_none` | ROC-AUC of the mitigated arm minus the unmitigated arm of the same cell. `extra_qpu_min` is the additional QPU time of the mitigated arm |
| `tuned_minus_asrun` | ROC-AUC of the kernel support vector machine refitted post hoc with hyperparameters chosen by threefold cross-validation on the training units (on the stored hardware kernel for FQK, on the standardised hardware features for PQK) minus the as-run classifier with fixed default settings. `auc_tuned` is the ROC-AUC of the tuned classifier |
| `hardware_minus_classical_fqk` | hardware FQK/angle minus its closed form on the same executed inputs, with the same 16 Nystrom landmarks (`b` = `closed-form nystrom`) or exact on all training units (`closed-form exact`). `auc_b` is the ROC-AUC of the closed-form model |
| `hardware_minus_classical_pqk` | hardware PQK/angle minus the same fixed-default RBF support vector machine on the closed-form projected features of the executed inputs (`b` = `closed-form features`). `auc_b` is the ROC-AUC of the closed-form model |

`p_bh` is the Benjamini-Hochberg adjusted `p_boot` within one family, over the rows with `primary`
true: the `rbf_svm` rows of the four baseline families, every row of
`trex_minus_none` and `hardware_minus_classical_pqk`, and the Nystrom rows of
`hardware_minus_classical_fqk`. The `tuned_minus_asrun` rows are not adjusted (`p_bh` empty).

### `screen/screen_v2/` (written by `screen.py`)

File prefix `<mode>_vqcsv` for the device-noise (`noisy`) and noiseless (`sv`) screens with the
VQC trained noiselessly on binary cross-entropy, and `sv_vqcsv_legacy` for the noiseless screen of
the VQC with the one-sided objective (`screen.py --vqc-loss one-sided`).
`*_units.jsonl` holds one line per (task, model, encoding, seed) unit, `*_raw.csv` the same as a
table (`roc_auc`, `per_label_auc`, `depth`, `n_2q`, `phys_qubits`, `runtime_s`, `roc_auc_noent`,
`ent_sensitivity`), `*_ranked.csv` the seed means (`auc_mean`, `auc_std`, `auc_noent_mean`), and
`*_promotion.json` the best encoding per model family and task with its margin to the next.
`vqc_histories/*.json` store each VQC training run (`history`, `fit`, `n_weights`, `test_auc`).

### `baselines_<task>_<pool>.csv` (written by `run_baselines.py`)

Classical baselines on the capped and full pools with each task's split protocol (apnea: train
recordings versus test recordings, PTB-XL: folds 1 to 8 versus 9 and 10, WESAD: leave one
subject out). Columns per task:

| Files | Columns |
|-------|---------|
| `baselines_apnea_capped.csv`, `baselines_apnea_full.csv` | `balanced_acc`, `macro_f1`, `roc_auc`, `pr_auc`, `ece`, `brier`, `roc_auc_lo`, `roc_auc_hi` (95% percentile bootstrap interval over test minutes, 1000 resamples), `model` |
| `baselines_ptbxl_capped.csv`, `baselines_ptbxl_full.csv` | `macro_f1`, `roc_auc`, `pr_auc` (macro averages over the five superclasses), `model` |
| `baselines_wesad_capped.csv` | `model`, `balanced_acc` (mean over the held-out subjects), `balanced_acc_std` (standard deviation over subjects), `n_subjects`, `balanced_acc_lo`, `balanced_acc_hi` (2.5th and 97.5th percentiles of the per-subject values) |
