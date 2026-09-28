"""Generate summary tables (tables/*.csv) and, when manuscript/ exists, the numeric LaTeX tables of the paper.

Paper tables       manuscript/tables/tab_screen.tex (tab:screen), tab_hwmain.tex (tab:hwmain), and tab_counts.tex
                   (tab:counts), written only when manuscript/ exists. The paper reports the IBM Phoenix hardware
                   results on the full test pools.
Summary tables     tables/*.csv, generated locally from data/results/ (not part of the released repository), covering
                   the IBM Phoenix hardware cells on the full test pools. data/README.md documents the underlying
                   result files. Rows are sorted and floats written with one fixed format, so
                   rebuilding from unchanged results gives identical files.
No number in these tables is typed by hand, and a missing input raises instead of silently dropping a table.
Usage: uv run python experiments/make_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
A = RES / "analysis_v2"
MS = ROOT / "manuscript"
OUT = MS / "tables"                               # LaTeX tables of the paper
REPO = ROOT / "tables"                            # CSV tables of the repository
PHOENIX = "phoenix_2026-09"
DEVICE = {PHOENIX: "ibm_phoenix"}
PHX_POOL = A / f"{PHOENIX}_pool"
# (run tag, scope, analysis directory) of the analysed hardware evaluation
SCOPES = [(PHOENIX, "pool", PHX_POOL)]
TASKS = ("apnea", "ptbxl")
TASKHEAD = {"apnea": "Apnea (T2)", "ptbxl": "PTB-XL (T1)"}
MAIN_TABLES = ["tab_screen", "tab_hwmain", "tab_counts"]
REPO_CSV = ["hardware_cells.csv", "paired_differences.csv", "matched_baselines.csv", "tuned_kernels.csv",
            "kernel_spectra.csv", "kernel_agreement.csv", "hardware_jobs.csv", "screen.csv"]
FLOAT = "%.6g"                                    # one float format for every repository table

# ----------------------------------------------------------------------------- cell names
ORDER = ["fqk", "pqk", "pqk-zz", "vqc-sv", "vqc-noisy", "vqc-sv-noent", "vqc"]
BASE = {"fqk": "FQK", "pqk": "PQK", "pqk-zz": "PQK", "vqc": "VQC", "vqc-sv": "VQC", "vqc-noisy": "VQC",
        "vqc-sv-noent": "VQC"}
ENC = {"angle": "angle", "zz": "ZZ", "reuploading": "re-uploading"}
QUALIFIER = {"vqc-noisy": "noise-aware training", "vqc-sv-noent": "without two-qubit gates"}
OBJECTIVE = {"vqc-sv": "binary cross-entropy", "vqc-noisy": "binary cross-entropy", "vqc-sv-noent": "binary cross-entropy"}
TRAINING = {"vqc-sv": "noiseless simulation",
            "vqc-noisy": "simulation under the IBM Phoenix noise model", "vqc-sv-noent": "noiseless simulation"}
BASELINES = ["logreg", "rbf_svm", "xgboost", "mlp"]
FAMILIES = ["qml_vs_baseline", "qml_vs_baseline_trex", "qml_vs_baseline_ba", "qml_vs_baseline_ba_trex",
            "trex_minus_none", "tuned_minus_asrun", "hardware_minus_classical_fqk", "hardware_minus_classical_pqk"]


def cell_label(model: str, encoding: str) -> str:
    """Encoding-correct cell name, e.g. 'VQC/angle, noise-aware training'."""
    s = f"{BASE[model]}/{ENC[encoding]}"
    return s + (f", {QUALIFIER[model]}" if model in QUALIFIER else "")


def mitigation(model: str, mitig: str) -> str:
    """What the mitigated arm applies: TREX on the Estimator circuits (PQK, VQC), measurement twirling on the
    Sampler circuits (FQK), as configured in hw_run2.make_primitives."""
    if mitig == "none":
        return "none"
    return "measurement twirling" if model == "fqk" else "TREX"


def order_key(model):
    return ORDER.index(model)


def _parse_a(a: str) -> tuple[str, str]:
    """'fqk/none', 'fqk/trex hardware', 'vqc-sv/trex', 'pqk/none tuned' -> (model, mitig)."""
    model, rest = a.split("/", 1)
    return model, rest.split()[0]


# ----------------------------------------------------------------------------- formatting and I/O
def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _sb(x, d=3):
    """Signed bound. A bound that rounds to zero keeps its first significant digit (never prints -0.000)."""
    if x != 0 and round(abs(x), d) == 0:
        d = int(-np.floor(np.log10(abs(x))))
    return f"{x:+.{d}f}"


def sgn(x):
    return f"${x:+.3f}$"


def sci(lo, hi):
    return f"$[{_sb(lo)}, {_sb(hi)}]$"


def br(lo, hi):
    return f"[{lo:.2f}, {hi:.2f}]"


def one(df: pd.DataFrame, what: str, **cond) -> pd.Series:
    """Exactly one matching row, or raise (missing inputs must never be skipped silently)."""
    m = np.ones(len(df), bool)
    for k, v in cond.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise LookupError(f"{what}: expected one row for {cond}, found {int(m.sum())}")
    return df[m].iloc[0]


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _w(name, s):
    if not MS.is_dir():                           # repository without the paper sources: CSV tables only
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(s, encoding="utf8"); print("  wrote", f"manuscript/tables/{name}.tex")


WRITTEN_CSV: set[str] = set()


def _csv(name: str, rows: list[dict], sort: list[str]):
    """Write a repository table: rows sorted by the given (already ordinal) columns, which are then dropped."""
    REPO.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(sort, kind="mergesort").drop(columns=sort)
    df.to_csv(REPO / name, index=False, float_format=FLOAT, lineterminator="\n")
    WRITTEN_CSV.add(name)
    print("  wrote", f"tables/{name}", f"({len(df)} rows)")


def _ord(tag, scope, task=None, model=None, mitig=None):
    """Ordinal sort columns shared by the summary tables."""
    o = {"_d": [t for t, _, _ in SCOPES].index(tag), "_s": 0 if scope == "pool" else 1}
    if task is not None:
        o["_t"] = TASKS.index(task)
    if model is not None:
        o["_m"] = order_key(model)
    if mitig is not None:
        o["_x"] = 0 if mitig == "none" else 1
    return o


# ----------------------------------------------------------------------------- tab:counts (appendix)
def tab_counts():
    d = json.loads((A / "data_description.json").read_text())["subsets"]
    # the hardware models are trained on these sets and scored on the pool rows (checked against the Phoenix cells)
    cells = read(RES / "hw" / PHOENIX / "cells.csv")
    for task in TASKS:
        for base, models in (("fqk", ["fqk"]), ("pqk", ["pqk", "pqk-zz"]), ("vqc", ["vqc-sv", "vqc-noisy", "vqc-sv-noent"])):
            for m in models:
                c = cells[(cells.task == task) & (cells.model == m)]
                if len(c) and not (c.n_train == d[f"hardware/{task}/{base}"]["train"]["n"]).all():
                    raise ValueError(f"IBM Phoenix {task} {m} training size differs from hardware/{task}/{base}")
                if len(c) and not (c.n_pool == d[f"pool/{task}"]["test"]["n"]).all():
                    raise ValueError(f"IBM Phoenix {task} {m} is not scored on the full test pool")

    def cls(x):
        if "pos" in x:
            return f"{x['neg']}/{x['pos']}"
        return "/".join(str(x[c]) for c in ("NORM", "MI", "STTC", "CD", "HYP"))

    def train(v):
        return f"{v['n']} ({v['n_groups']}) & {cls(v)}"

    def test(v):
        return f"{v['n']} ({v['n_groups']}, {v['per_group_median']:.0f}/{v['per_group_max']}) & {cls(v)}"
    pool_test = "\\multicolumn{2}{l}{full test pool}"
    spec = [("pool/apnea", "Apnea pools", "both"), ("hardware/apnea/fqk", "Apnea FQK training set", "train"),
            ("hardware/apnea/pqk", "Apnea PQK training set", "train"), ("hardware/apnea/vqc", "Apnea VQC training set", "train"),
            ("screen/apnea/seed0", "Apnea screen (seed 0)", "both"),
            ("pool/ptbxl", "PTB-XL pools", "both"), ("hardware/ptbxl/fqk", "PTB-XL FQK training set", "train"),
            ("hardware/ptbxl/pqk", "PTB-XL PQK training set", "train"), ("hardware/ptbxl/vqc", "PTB-XL VQC training set", "train"),
            ("screen/ptbxl/seed0", "PTB-XL screen (seed 0)", "both"),
            ("pool/wesad", "WESAD pools (screen split)", "both"), ("screen/wesad/seed0", "WESAD screen (seed 0)", "both")]
    rows = []
    for key, lab, kind in spec:
        v = d[key]
        tr = train(v["train"]) if kind in ("both", "train") else "-- & --"
        te = test(v["test"]) if kind in ("both", "test") else pool_test
        rows.append(f"{lab} & {tr} & {te} \\\\")
    s = ("\\begin{table}[h]\n\\caption{Sample sizes, with the clusters (Apnea-ECG recordings, PTB-XL patients, WESAD subjects) in "
         "parentheses and, for test sets, the median and maximum units per cluster. Classes are negative/positive, or "
         "NORM/MI/STTC/CD/HYP for PTB-XL. Hardware models are scored on the full test pool of their task.}\\label{tab:counts}\n"
         "\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
         "Set & Train $n$ (clusters) & Train classes & Test $n$ (clusters, median/max per cluster) & Test classes \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    _w("tab_counts", s)


# ----------------------------------------------------------------------------- tab:screen (results)
SCREEN = RES / "screen" / "screen_v2"


def tab_screen():
    noisy = read(SCREEN / "noisy_vqcsv_ranked.csv")
    sv = read(SCREEN / "sv_vqcsv_ranked.csv")
    prom = json.loads((SCREEN / "noisy_vqcsv_promotion.json").read_text())
    promoted = {(p["task"], p["model"], p["encoding"]) for p in prom}
    cells = [("fqk", "angle"), ("pqk", "angle"), ("fqk", "zz"), ("pqk", "zz"), ("vqc", "angle"), ("vqc", "zz"), ("vqc", "reuploading")]
    lines = []
    for m, e in cells:
        row = [f"{m.upper()}/{ENC[e]}"]
        for t in ("apnea", "ptbxl", "wesad"):
            r = noisy[(noisy.task == t) & (noisy.model == m) & (noisy.encoding == e)]
            if not len(r):
                row += ["--", "--"]; continue
            r = r.iloc[0]
            cell = f"{r.auc_mean:.3f} $\\pm$ {r.auc_std:.3f}"
            if not np.isnan(r.auc_noent_mean):
                cell += f" ({r.auc_noent_mean:.2f})"
            if (t, m, e) in promoted:
                cell = f"\\textit{{{cell}}}"
            row.append(cell)
            q = sv[(sv.task == t) & (sv.model == m) & (sv.encoding == e)]
            row.append(f"{q.auc_mean.iloc[0]:.3f}" if len(q) else "--")
        lines.append(" & ".join(row) + " \\\\")
    phx = read(RES / "hw" / PHOENIX / "cells.csv")
    extra = sorted({(r.task, r.model.split("-")[0], r.encoding) for _, r in phx.iterrows() if not r.model.startswith("vqc")}
                   - promoted)
    if extra != [("ptbxl", "pqk", "angle")]:
        raise ValueError(f"the caption names PTB-XL PQK/angle as the only executed kernel cell outside the promotion, got {extra}")
    seeds = read(SCREEN / "noisy_vqcsv_raw.csv").seed.nunique()
    head = "Cell & \\multicolumn{2}{c}{apnea (T2)} & \\multicolumn{2}{c}{PTB-XL (T1)} & \\multicolumn{2}{c}{WESAD (T3)} \\\\\n" \
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & noisy & noiseless & noisy & noiseless & noisy & noiseless \\\\\n"
    s = ("\\begin{table*}[t]\n\\caption{Simulation screen, ROC-AUC mean $\\pm$ standard deviation over "
         f"{['zero', 'one', 'two', 'three', 'four', 'five'][seeds]} seeds. "
         "Noisy columns use the noise model of an IBM Heron r3 processor (June 2026 calibration), with the entanglement-removed control in "
         "parentheses, and noiseless columns exact statevectors. The noisy FQK column uses a 32-landmark Nystr\\\"om kernel and the "
         "noiseless column the exact kernel. Italic cells are promoted to hardware.}"
         "\\label{tab:screen}\n\\centering\n\\small\n"
         "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n" + head + "\\midrule\n" + "\n".join(lines) +
         "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    _w("tab_screen", s)


# ----------------------------------------------------------------------------- tab:hwmain (results)
HWMAIN_ROWS = {"apnea": ["fqk", "pqk", "vqc-sv", "vqc-noisy", "vqc-sv-noent"],
               "ptbxl": ["fqk", "pqk", "pqk-zz", "vqc-sv", "vqc-noisy", "vqc-sv-noent"]}


def tab_hwmain():
    """IBM Phoenix on the full test pools, unmitigated arms. Each model takes two lines, the estimates on the first and
    the 95% intervals in brackets on the second (next to the qualifier of the cell name)."""
    cm = read(PHX_POOL / "cells_metrics.csv")
    bl = read(PHX_POOL / "baselines.csv")
    pa = read(PHX_POOL / "paired.csv")
    lines = []
    for task in TASKS:
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{TASKHEAD[task]}}}}} \\\\")
        for model in HWMAIN_ROWS[task]:
            p = one(cm, "IBM Phoenix pool cells", task=task, model=model, mitig="none")
            b = one(bl, "IBM Phoenix pool baselines", task=task, cell=f"{model}/none", baseline="rbf_svm")
            d = one(pa, "paired ROC-AUC", family="qml_vs_baseline", task=task, a=f"{model}/none", b="rbf_svm")
            e = one(pa, "paired balanced accuracy", family="qml_vs_baseline_ba", task=task, a=f"{model}/none", b="rbf_svm")
            has_trex = ((cm.task == task) & (cm.model == model) & (cm.mitig == "trex")).any()
            t = one(pa, "mitigation", family="trex_minus_none", task=task, a=f"{model}/trex", b=f"{model}/none") if has_trex else None
            if not has_trex and ((pa.family == "trex_minus_none") & (pa.task == task) & (pa.a == f"{model}/trex")).any():
                raise ValueError(f"{task} {model}: mitigation difference without a mitigated arm")
            l1, _, l2 = cell_label(model, p.encoding).partition(", ")
            top = [l1, f3(p.roc_auc), f3(p.bal_acc), f3(p.ece), f"{p.qpu_min:.2f}", f3(b.roc_auc), sgn(d["diff"]), sgn(e["diff"]),
                   sgn(t["diff"]) if t is not None else "--"]
            bot = [l2, br(p.auc_lo_cluster, p.auc_hi_cluster), "", "", "", "", sci(d.lo, d.hi), sci(e.lo, e.hi),
                   sci(t.lo, t.hi) if t is not None else ""]
            lines.append(" & ".join(top) + " \\\\")
            lines.append(" & ".join(bot) + " \\\\[1pt]")
        lines.append("\\midrule")
    lines = lines[:-1]
    size = {}
    for task in TASKS:                              # the caption states one pool size per task
        c = cm[cm.task == task]
        if c.n_test.nunique() != 1 or c.n_clusters.nunique() != 1:
            raise ValueError(f"IBM Phoenix {task} cells are not all scored on one pool")
        size[task] = (int(c.n_test.iloc[0]), int(c.n_clusters.iloc[0]))
    from experiments.analysis_hw import B as nboot               # bootstrap resamples drawn by the analysis
    if (pa.n_boot > nboot).any():
        raise ValueError(f"paired.csv reports more resamples than analysis_hw.B = {nboot}")
    s = ("\\begin{table*}[t]\n\\caption{Hardware results on IBM Phoenix on the full test pools "
         f"({size['apnea'][0]} apnea minutes, {size['ptbxl'][0]} PTB-XL records), unmitigated arm. Brackets give 95\\% "
         "cluster-bootstrap intervals. $\\Delta$ is the paired difference to the RBF-SVM trained on the same units and tested on the "
         "same executed inputs, and the mitigation column gives the mitigated minus the unmitigated arm. Balanced accuracy and ECE "
         "are averaged over superclasses for PTB-XL.}\\label{tab:hwmain}\n"
         "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\sbox0{%\n\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
         " & \\multicolumn{4}{c}{Quantum model} & RBF-SVM & \\multicolumn{2}{c}{Quantum $-$ RBF-SVM} & Mitigation \\\\\n"
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}\n"
         "Model & ROC-AUC & Bal.\\ acc. & ECE & QPU min & ROC-AUC & $\\Delta$ROC-AUC & $\\Delta$Bal.\\ acc. & $\\Delta$ROC-AUC \\\\\n"
         "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}%\n"
         "\\ifdim\\wd0>\\textwidth \\resizebox{\\textwidth}{!}{\\usebox0}\\else\\usebox0\\fi\n\\end{table*}\n")   # shrink, never enlarge
    _w("tab_hwmain", s)


# ----------------------------------------------------------------------------- summary tables
def _cellinfo(model, encoding):
    return {"cell": cell_label(model, encoding), "model": model, "encoding": encoding,
            "objective": OBJECTIVE.get(model, ""), "training": TRAINING.get(model, "")}


def repo_hardware_cells():
    rows = []
    for tag, scope, d in SCOPES:
        for _, r in read(d / "cells_metrics.csv").iterrows():
            rows.append({**_ord(tag, scope, r.task, r.model, r.mitig), "device": DEVICE[tag], "scope": scope, "task": r.task,
                         **_cellinfo(r.model, r.encoding), "mitig": r.mitig, "mitigation": mitigation(r.model, r.mitig),
                         "roc_auc": r.roc_auc, "auc_lo_cluster": r.auc_lo_cluster, "auc_hi_cluster": r.auc_hi_cluster,
                         "auc_lo_iid": r.auc_lo_iid, "auc_hi_iid": r.auc_hi_iid, "roc_auc_platt": r.roc_auc_platt,
                         "pr_auc": r.pr_auc, "bal_acc": r.bal_acc, "f1": r.f1, "threshold": r.threshold, "ece": r.ece,
                         "brier": r.brier, "n_train": int(r.n_train), "n_test": int(r.n_test), "n_clusters": int(r.n_clusters),
                         "parameter_sets": int(r.circuits), "qpu_min": r.qpu_min, "bell": r.bell, "backend": r.backend})
    _csv("hardware_cells.csv", rows, ["_d", "_s", "_t", "_m", "_x"])


def repo_paired():
    rows = []
    need = {"qml_vs_baseline", "qml_vs_baseline_ba", "trex_minus_none", "tuned_minus_asrun", "hardware_minus_classical_fqk",
            "hardware_minus_classical_pqk", "qml_vs_baseline_trex", "qml_vs_baseline_ba_trex"}
    for tag, scope, d in SCOPES:
        cm = read(d / "cells_metrics.csv")
        pa = read(d / "paired.csv")
        if need - set(pa.family):
            raise LookupError(f"{d / 'paired.csv'} lacks the families {sorted(need - set(pa.family))} (rerun analysis_hw.py)")
        for _, r in pa.iterrows():
            model, mitig = _parse_a(r.a)
            enc = one(cm, f"{tag} cells", task=r.task, model=model, mitig=mitig).encoding
            rows.append({**_ord(tag, scope, r.task, model, mitig), "_f": FAMILIES.index(r.family),
                         "_b": BASELINES.index(r.b) if r.b in BASELINES else 9, "_bb": r.b,
                         "device": DEVICE[tag], "scope": scope, "task": r.task, "family": r.family,
                         "metric": "bal_acc" if r.family.startswith("qml_vs_baseline_ba") else "roc_auc",
                         "cell": cell_label(model, enc), "model": model, "mitig": mitig, "a": r.a, "b": r.b,
                         "diff": r["diff"], "lo": r.lo, "hi": r.hi, "lo_iid": r.get("lo_iid"), "hi_iid": r.get("hi_iid"),
                         "p_gt0": r.p_gt0, "p_boot": r.p_boot, "primary": bool(r.primary), "p_bh": r.p_bh,
                         "n_boot": int(r.n_boot), "auc_a": r.auc_tuned if r.family == "tuned_minus_asrun" else np.nan,
                         "auc_b": r.auc_b, "extra_qpu_min": r.extra_qpu_min, "proba_corr": np.nan, "n": int(one(
                             cm, f"{tag} cells", task=r.task, model=model, mitig=mitig).n_test)})
    _csv("paired_differences.csv", rows, ["_d", "_s", "_t", "_f", "_m", "_x", "_b", "_bb"])


def repo_baselines():
    rows = []
    for tag, scope, d in SCOPES:
        cm = read(d / "cells_metrics.csv")
        for _, r in read(d / "baselines.csv").iterrows():
            model, mitig = _parse_a(r.cell)
            enc = one(cm, f"{tag} cells", task=r.task, model=model, mitig=mitig).encoding
            rows.append({**_ord(tag, scope, r.task, model, mitig), "_b": BASELINES.index(r.baseline),
                         "device": DEVICE[tag], "scope": scope, "task": r.task, "inputs_of": cell_label(model, enc),
                         "model": model, "mitig": mitig, "baseline": r.baseline, "roc_auc": r.roc_auc,
                         "auc_lo_cluster": r.auc_lo_cluster, "auc_hi_cluster": r.auc_hi_cluster, "auc_lo_iid": r.auc_lo_iid,
                         "auc_hi_iid": r.auc_hi_iid, "roc_auc_platt": r.roc_auc_platt, "pr_auc": r.pr_auc, "bal_acc": r.bal_acc,
                         "f1": r.f1, "threshold": r.threshold, "ece": r.ece, "brier": r.brier, "n_test": int(r.n_test),
                         "n_clusters": int(r.n_clusters)})
    _csv("matched_baselines.csv", rows, ["_d", "_s", "_t", "_m", "_x", "_b"])


def repo_tuned():
    from experiments.models import PQK_GRID                      # the searched grids, as used by analysis_hw
    from experiments.analysis_hw import C_GRID
    grid = {"fqk": "C in {" + ", ".join(map(str, C_GRID["C"])) + "}",
            "pqk": "C in {" + ", ".join(map(str, PQK_GRID["C"])) + "}, gamma in {" + ", ".join(map(str, PQK_GRID["gamma"])) + "}"}
    rows = []
    for tag, scope, d in SCOPES:
        cm = read(d / "cells_metrics.csv")
        t = read(d / "paired.csv")
        t = t[t.family == "tuned_minus_asrun"]
        if not len(t):
            raise LookupError(f"no tuned_minus_asrun rows in {d / 'paired.csv'}")
        for _, r in t.iterrows():
            model, mitig = _parse_a(r.a)
            c = one(cm, f"{tag} cells", task=r.task, model=model, mitig=mitig)
            if abs(r.auc_tuned - r["diff"] - c.roc_auc) > 1e-9:
                raise ValueError(f"tuned row inconsistent with cells_metrics for {tag} {scope} {r.task} {r.a}")
            rows.append({**_ord(tag, scope, r.task, model, mitig), "device": DEVICE[tag], "scope": scope, "task": r.task,
                         "cell": cell_label(model, c.encoding), "model": model, "mitig": mitig,
                         "search_grid": grid["fqk" if model == "fqk" else "pqk"], "auc_as_run": c.roc_auc,
                         "auc_tuned": r.auc_tuned, "diff": r["diff"], "lo": r.lo, "hi": r.hi, "p_gt0": r.p_gt0,
                         "p_boot": r.p_boot, "n_boot": int(r.n_boot),
                         "reselected": bool(r["diff"] == 0 and r.lo == 0 and r.hi == 0)})
    _csv("tuned_kernels.csv", rows, ["_d", "_s", "_t", "_m", "_x"])


def _kernel_name(k: str) -> tuple[str, float]:
    """(plain kernel name, RBF bandwidth or NaN) without the noiseless/hardware tag of analysis_hw."""
    g = re.search(r"gamma=([0-9.eE+-]+)", k)
    if k.startswith("exact"):
        return "exact (closed form)", np.nan
    if k.startswith("Nystrom"):
        return f"Nystrom, L={re.search(r'L=(\d+)', k).group(1)}", np.nan
    if k.startswith("RBF"):
        return "RBF on standardised projected features", float(g.group(1))
    if k.startswith("linear Gram"):
        return "linear Gram of raw projected features", np.nan
    raise ValueError(f"unknown kernel {k!r}")


def repo_kernels():
    """Spectra of the training kernels (they do not depend on the test set)."""
    rows = []
    for tag, d in ((PHOENIX, PHX_POOL),):
        for _, r in read(d / "kernels.csv").iterrows():
            name, gamma = _kernel_name(r.kernel)
            rows.append({"_d": 0, "_t": TASKS.index(r.task), "_k": 0 if r.model.startswith("FQK") else 1,
                         "_n": ["exact", "Nystrom", "RBF", "linear"].index(name.split()[0].rstrip(",")),
                         "_src": 0 if r.source == "noiseless" else 1,
                         "device": DEVICE[tag], "task": r.task, "cell": r.model, "kernel": name, "source": r.source,
                         "gamma": gamma, "n": int(r.n), "eff_rank": r.eff_rank, "num_rank": int(r.num_rank), "k95": int(r.k95),
                         "kta_centered": r.kta_centered, "lambda_min": r.lambda_min, "condition": r["condition"]})
    _csv("kernel_spectra.csv", rows, ["_d", "_t", "_k", "_n", "_src"])


def repo_kernel_agreement():
    rows = []
    for tag, scope, d in SCOPES:
        fa = read(d / "kernel_agreement.csv")
        pf = read(d / "pqk_feature_agreement.csv")
        base = lambda r: {**_ord(tag, scope, r.task), "device": DEVICE[tag], "scope": scope, "task": r.task}   # noqa: E731
        for _, r in fa.iterrows():
            for j, (b, block) in enumerate((("LL", "landmarks"), ("trL", "training"), ("teL", "test"))):
                rows.append({**base(r), "_m": 0, "_x": 0 if r.mitig == "none" else 1, "_j": j, "_q": 0,
                             "cell": "FQK/angle", "mitig": r.mitig, "block": block, "quantity": "kernel entries",
                             "pearson_r": r[f"{b}_corr"], "mae": r[f"{b}_mae"], "slope": r[f"{b}_slope"], "sd_measured": np.nan})
        for _, r in pf.iterrows():
            for j, (side, block) in enumerate((("tr", "training"), ("te", "test"))):
                for q, name in enumerate(("X", "Z")):
                    rows.append({**base(r), "_m": 1, "_x": 0 if r.mitig == "none" else 1, "_j": j + 1, "_q": q,
                                 "cell": "PQK/angle", "mitig": r.mitig, "block": block, "quantity": f"<{name}_i>",
                                 "pearson_r": r[f"{side}{name}_corr"], "mae": r[f"{side}{name}_mae"], "slope": np.nan,
                                 "sd_measured": np.nan})
                rows.append({**base(r), "_m": 1, "_x": 0 if r.mitig == "none" else 1, "_j": j + 1, "_q": 2,
                             "cell": "PQK/angle", "mitig": r.mitig, "block": block, "quantity": "<Y_i>",
                             "pearson_r": np.nan, "mae": np.nan, "slope": np.nan, "sd_measured": r[f"{side}Y_sd"]})
    _csv("kernel_agreement.csv", rows, ["_d", "_s", "_t", "_m", "_x", "_j", "_q"])


def repo_jobs():
    rows = []
    for tag in (PHOENIX,):
        for _, r in read(RES / "hw" / tag / "cells.csv").iterrows():
            n_exec = r.get("n_pool", np.nan)
            rows.append({**_ord(tag, "pool", r.task, r.model, r.mitig), "device": DEVICE[tag], "task": r.task,
                         **_cellinfo(r.model, r.encoding), "mitig": r.mitig, "mitigation": mitigation(r.model, r.mitig),
                         "parameter_sets": int(r.circuits), "shots": int(r.shots), "quantum_seconds": int(r.quantum_seconds),
                         "qpu_min": r.qpu_min, "bell": r.bell,
                         "two_qubit_gates": np.nan if pd.isna(r.get("two_qubit")) else int(r.two_qubit),
                         "depth": np.nan if pd.isna(r.get("depth")) else int(r.depth), "n_train": int(r.n_train),
                         "n_test_executed": int(r.n_test if pd.isna(n_exec) else n_exec),
                         "session": r.session, "jobs": r.jobs, "started": r.get("started", ""), "backend": r.backend})
    _csv("hardware_jobs.csv", rows, ["_d", "_s", "_t", "_m", "_x"])


def repo_screen():
    prom = {(p["task"], p["model"], p["encoding"]) for p in json.loads((SCREEN / "noisy_vqcsv_promotion.json").read_text())}
    rows = []
    for j, (f, mode, objective) in enumerate((("noisy_vqcsv", "noisy", "binary cross-entropy"),
                                              ("sv_vqcsv", "noiseless", "binary cross-entropy"),
                                              ("sv_vqcsv_legacy", "noiseless", "one-sided cross-entropy"))):
        rank, raw = read(SCREEN / f"{f}_ranked.csv"), read(SCREEN / f"{f}_raw.csv")
        for _, r in rank.iterrows():
            nseed = raw[(raw.task == r.task) & (raw.model == r.model) & (raw.encoding == r.encoding)].seed.nunique()
            kern = {"fqk": "Nystrom, L=32" if mode == "noisy" else "exact", "pqk": "RBF on standardised projected features"}
            rows.append({"_j": j, "_t": ["apnea", "ptbxl", "wesad"].index(r.task), "_m": ["fqk", "pqk", "vqc"].index(r.model),
                         "_e": list(ENC).index(r.encoding), "task": r.task, "cell": f"{r.model.upper()}/{ENC[r.encoding]}",
                         "model": r.model, "encoding": r.encoding, "mode": mode,
                         "noise_model": "heron_r3_2026-06-11" if mode == "noisy" else "none",
                         "objective": objective if r.model == "vqc" else "", "kernel": kern.get(r.model, ""),
                         "n_seeds": int(nseed), "auc_mean": r.auc_mean, "auc_std": r.auc_std,
                         "auc_noent_mean": r.auc_noent_mean, "two_qubit_gates": r.n_2q, "runtime_s": r.runtime_s,
                         "promoted": (r.task, r.model, r.encoding) in prom if mode == "noisy" else ""})
    _csv("screen.csv", rows, ["_j", "_t", "_m", "_e"])


# ----------------------------------------------------------------------------- driver
GENERATORS = [tab_counts, tab_screen, tab_hwmain, repo_hardware_cells, repo_paired, repo_baselines, repo_tuned,
              repo_kernels, repo_kernel_agreement, repo_jobs, repo_screen]


def _check_inputs():
    """Every analysis file this script reads, listed at once before anything is written."""
    need = [A / "data_description.json"]
    for _, _, d in SCOPES:
        need += [d / f for f in ("cells_metrics.csv", "baselines.csv", "paired.csv", "kernels.csv", "kernel_agreement.csv",
                                 "pqk_feature_agreement.csv")]
    miss = [p.relative_to(ROOT).as_posix() for p in need if not p.exists()]
    if miss:
        raise SystemExit("missing inputs (run the analysis stage first):\n  " + "\n  ".join(miss))


def _clean_stale():
    """Delete generated files this version no longer writes: tab_*.tex outside MAIN_TABLES and every CSV in tables/
    that this run did not write (a renamed table leaves no orphan behind)."""
    if WRITTEN_CSV != set(REPO_CSV):              # every generator must have run before anything is deleted
        raise RuntimeError(f"CSV set mismatch, not cleaning: wrote {sorted(WRITTEN_CSV)}, expected {sorted(REPO_CSV)}")
    for f in OUT.glob("tab_*.tex"):
        if f.stem not in MAIN_TABLES:
            f.unlink(); print("  removed stale", f.relative_to(ROOT).as_posix())
    for f in REPO.glob("*.csv"):
        if f.name not in WRITTEN_CSV:
            f.unlink(); print("  removed stale", f.relative_to(ROOT).as_posix())


def main():
    sys.path.insert(0, str(ROOT))                 # repo_tuned imports the searched grids from experiments.*
    _check_inputs()
    for fn in GENERATORS:
        fn()
    _clean_stale()


if __name__ == "__main__":
    main()
