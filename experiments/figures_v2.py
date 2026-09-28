"""Paper figures, generated only from the result files, so every plotted value has one source. Every hardware series is
IBM Phoenix.

  fig_results.pdf      2x2. (a,b) spectra of the training kernels the models use (apnea, PTB-XL), noiseless and
                       measured on IBM Phoenix, with the effective rank and the centered alignment A in the legend;
                       (c) PTB-XL per-superclass ROC-AUC on the full test pool; (d) apnea ROC-AUC of random test subsets
                       of the full pool with the trained model fixed (pool_subsets.pool_subset_draws)
  fig_calibration.pdf  reliability diagram of the apnea test pool, marker area proportional to the minutes in a bin,
                       open markers for bins below pool_subsets.MIN_BIN minutes
  fig_vqc.pdf          VQC learning curves (appendix)

Sizes follow the iopjournal text block (153 mm), so no font prints below 6.5 pt. Colours are a colour-vision-deficiency
checked categorical palette. The PDFs carry no creation date, so rebuilding from unchanged results gives identical files.
Output goes to figures/ and, when it exists, to manuscript/.
Usage: uv run python experiments/figures_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from experiments import stats as S                                   # noqa: E402
from experiments import kernel_diag as KD                            # noqa: E402
from experiments.analysis_hw import _sv_projected                    # noqa: E402
from experiments.config import TASKS                                 # noqa: E402
from experiments.pool_subsets import MIN_BIN, pool_subset_draws      # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

RES = ROOT / "data" / "results"
OUTS = ([ROOT / "manuscript"] if (ROOT / "manuscript").is_dir() else []) + [ROOT / "figures"]
PHOENIX = "phoenix_2026-09"
DEV = "IBM Phoenix"
POOL = RES / "analysis_v2" / f"{PHOENIX}_pool"                      # full test pools
CLASSES = TASKS["ptbxl"]["classes"]
TEXTWIDTH_IN = 153 / 25.4                                            # iopjournal \textwidth
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.handlelength": 1.8,
                     "axes.spines.top": False, "axes.spines.right": False})
# categorical palette, checked for colour-vision-deficiency separation (dataviz validator)
C_FQK, C_PQK, C_VQC, C_VQCN = "#1f5fa8", "#e08214", "#2a9d8f", "#9b4f96"
TASKLAB = {"apnea": "apnea", "ptbxl": "PTB-XL", "wesad": "WESAD"}
PDF_META = {"CreationDate": None, "ModDate": None}


def _panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".rstrip(), loc="left", fontweight="bold", fontsize=7.5, pad=3)


def _save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    for o in OUTS:
        o.mkdir(parents=True, exist_ok=True); fig.savefig(o / name, metadata=PDF_META)
    plt.close(fig); print("  saved", name)


def _read(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _pred() -> pd.DataFrame:
    return _read(RES / "hw" / PHOENIX / "predictions.csv")


# ----------------------------------------------------------------------------- (a,b) spectra
def spectra_curves(task):
    mdir = RES / "hw" / PHOENIX / "matrices"
    mf = np.load(mdir / f"{task}_fqk_none.npz"); mp = np.load(mdir / f"{task}_pqk_none.npz")
    for z, keys in ((mf, ("Xtr_s", "landmarks", "G_tr")), (mp, ("Xtr_s", "F_tr"))):
        miss = [k for k in keys if k not in z.files]
        if miss:
            raise KeyError(f"{PHOENIX} {task}: matrices lack {miss}")
    nq = TASKS[task]["qubits"]
    Xf, L = mf["Xtr_s"], mf["landmarks"]
    K_exact = KD.fqk_angle_closed_form(Xf, Xf)
    K_nys, _ = KD.nystrom(KD.fqk_angle_closed_form(Xf, L), KD.fqk_angle_closed_form(L, L))
    K_rbf_sv, _ = KD.rbf_gram(StandardScaler().fit_transform(_sv_projected(mp["Xtr_s"], nq)))
    K_rbf_hw, _ = KD.rbf_gram(StandardScaler().fit_transform(mp["F_tr"]))
    return [("FQK exact", "exact", K_exact, C_FQK, "-"),
            ("FQK Nyström", "Nystrom L=16 (noiseless)", K_nys, C_FQK, "--"),
            ("FQK Nyström, device", "Nystrom L=16 (hardware)", mf["G_tr"], C_FQK, ":"),
            ("PQK RBF", "RBF on standardised features (noiseless", K_rbf_sv, C_PQK, "-"),
            ("PQK RBF, device", "RBF on standardised features (hardware", K_rbf_hw, C_PQK, ":")]


def panel_spectra(ax, task, kd):
    for lab, key, K, col, ls in spectra_curves(task):
        w = np.clip(KD.spectrum(K), 0, None); w = w[w > 1e-12] / w.sum()
        r = kd[(kd.task == task) & (kd.kernel.str.startswith(key))]
        if len(r) != 1:
            raise LookupError(f"kernels.csv: expected one row for {task} {key}, found {len(r)}")
        ax.plot(np.arange(1, len(w) + 1), w, ls=ls, color=col, lw=1.2,
                label=f"{lab}  {r.eff_rank.iloc[0]:.1f} / {r.kta_centered.iloc[0]:.3f}")
    ax.set_yscale("log"); ax.set_xlabel("eigenvalue index"); ax.set_ylabel("normalised eigenvalue")
    lo = ax.get_ylim()[0]; ax.set_ylim(lo, 1.0)                      # headroom up to the largest possible value
    ax.legend(loc="upper right", frameon=False, labelspacing=0.15, handlelength=2.0, handletextpad=0.5,
              borderaxespad=0.1, fontsize=5.5, title="effective rank / alignment A", title_fontsize=5.5,
              alignment="right")


# ----------------------------------------------------------------------------- (c) PTB-XL per superclass
def panel_perclass(ax):
    """PTB-XL one-vs-rest ROC-AUC per superclass from the decision scores on the IBM Phoenix full pool."""
    from sklearn.metrics import roc_auc_score
    pred = _pred()
    width = 0.26; xs = np.arange(len(CLASSES))
    series = [("fqk", C_FQK, "FQK/angle"), ("pqk", C_PQK, "PQK/angle"), ("vqc-sv", C_VQC, "VQC/angle")]
    for k, (mdl, col, lab) in enumerate(series):
        d = pred[(pred.task == "ptbxl") & (pred.model == mdl) & (pred.mitig == "none") & (pred.split == "test")]
        if d.label.nunique() != len(CLASSES):
            raise LookupError(f"IBM Phoenix PTB-XL predictions for {mdl} lack labels")
        aucs = [roc_auc_score(d[d.label == c].y_true, d[d.label == c].score) for c in range(len(CLASSES))]
        ax.bar(xs + (k - 1) * width, aucs, width * 0.92, color=col, label=lab)
    ax.axhline(0.5, color="0.35", ls=":", lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(CLASSES); ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("ROC-AUC"); ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.0, borderaxespad=0.1,
              fontsize=6)


# ----------------------------------------------------------------------------- (d) test-size sensitivity
def panel_sensitivity(ax):
    """Apnea ROC-AUC of random test subsets of the IBM Phoenix pool (trained model fixed): mean and central 95% of
    the draws per subset size."""
    for mdl, col, lab in [("fqk", C_FQK, "FQK/angle"), ("pqk", C_PQK, "PQK/angle")]:
        draws, y, sc, _ = pool_subset_draws(mdl)
        ns = np.array(sorted(draws)); v = [draws[n] for n in ns]
        lo, hi = [np.percentile(x, 2.5) for x in v], [np.percentile(x, 97.5) for x in v]
        ax.fill_between(ns, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(ns, lo, color=col, lw=0.6, ls="--"); ax.plot(ns, hi, color=col, lw=0.6, ls="--")
        ax.plot(ns, [x.mean() for x in v], "-", color=col, lw=1.2, label=lab)
    ax.set_xscale("log")
    ticks = [20, 40, 80, 160, 320, 510]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.axhline(0.5, color="0.35", ls=":", lw=0.7)
    ax.set_xlabel("test minutes drawn from the pool"); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.1, labelspacing=0.15, fontsize=6)
    ax.set_ylim(0.4, 1.0)


def fig_results():
    kd = _read(POOL / "kernels.csv")
    fig, ax = plt.subplots(2, 2, figsize=(0.98 * TEXTWIDTH_IN, 4.35), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, hspace=0.06, wspace=0.06)
    panel_spectra(ax[0, 0], "apnea", kd); _panel(ax[0, 0], "a", "apnea training kernels")
    panel_spectra(ax[0, 1], "ptbxl", kd); _panel(ax[0, 1], "b", "PTB-XL training kernels")
    panel_perclass(ax[1, 0]); _panel(ax[1, 0], "c", "PTB-XL superclasses, full test pool")
    panel_sensitivity(ax[1, 1]); _panel(ax[1, 1], "d", "apnea, random test subsets")
    _save(fig, "fig_results.pdf", tight=False)


# ----------------------------------------------------------------------------- calibration
SIZE_PER_MINUTE = 0.45                   # marker area in pt^2 per test minute in the bin


def _reliability(ax, y, p, label, color):
    """Ten equal-width bins as stats.ece. Marker area is proportional to the minutes in the bin. Bins with fewer than
    MIN_BIN minutes are drawn open and left out of the connecting line, so a bin of one or two minutes cannot draw
    the curve to 0 or 1."""
    idx = np.minimum((p * 10).astype(int), 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    xs, ys, ns = map(np.asarray, (xs, ys, ns))
    full = ns >= MIN_BIN
    ax.plot(xs[full], ys[full], color=color, lw=1.0, zorder=2)
    ax.scatter(xs[full], ys[full], s=SIZE_PER_MINUTE * ns[full], color=color, edgecolors="white", lw=0.5, zorder=3)
    ax.scatter(xs[~full], ys[~full], s=max(SIZE_PER_MINUTE * 8, 4), facecolors="white", edgecolors=color, lw=0.7,
               zorder=3)
    return Line2D([], [], color=color, lw=1.0, marker="o", ms=3.5, label=label)


def fig_calibration():
    pred = _pred()
    fig, ax = plt.subplots(figsize=(0.46 * TEXTWIDTH_IN, 0.46 * TEXTWIDTH_IN), layout="constrained")
    ax.plot([0, 1], [0, 1], color="0.35", ls=":", lw=0.8)
    handles = []
    for mdl, col, lab in [("fqk", C_FQK, "FQK"), ("pqk", C_PQK, "PQK"), ("vqc-sv", C_VQC, "VQC"),
                          ("vqc-noisy", C_VQCN, "VQC, noise-aware")]:
        d = pred[(pred.task == "apnea") & (pred.model == mdl) & (pred.mitig == "none") & (pred.split == "test")]
        if not len(d):
            raise LookupError(f"no IBM Phoenix apnea predictions for {mdl}")
        handles.append(_reliability(ax, d.y_true.to_numpy(), d.proba.to_numpy(), lab, col))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed apnea frequency")
    sizes = [10, 50, 150]
    key = [ax.scatter([], [], s=SIZE_PER_MINUTE * n, color="0.5", edgecolors="white", lw=0.5) for n in sizes]
    ax.add_artist(ax.legend(key, [f"{n}" for n in sizes], title="minutes per bin", frameon=False, loc="upper left",
                            bbox_to_anchor=(0.0, 0.93), ncol=3, columnspacing=0.6, handletextpad=0.1, borderaxespad=0.1,
                            title_fontsize=6.5, alignment="left"))
    ax.legend(handles=handles, frameon=False, loc="lower right", borderaxespad=0.1, labelspacing=0.3, handlelength=1.5)
    _save(fig, "fig_calibration.pdf", tight=False)


# ----------------------------------------------------------------------------- VQC
def fig_vqc():
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.1))
    wdir = ROOT / "data" / "hardware" / PHOENIX
    for tr, col, lab in [("sv", C_VQC, "noiseless training"), ("noisy", C_VQCN, "noise-aware training")]:
        f = wdir / f"vqc_apnea_angle_{tr}_label0.npz"
        if not f.exists():
            raise FileNotFoundError(f)
        h = pd.DataFrame(json.loads(str(np.load(f)["history"])))
        ax[0].plot(h["iter"], h["train_loss"], color=col, lw=1, label=lab)
        hh = h.dropna(subset=["train_auc"])
        ax[1].plot(hh["iter"], hh["train_auc"], "-o", ms=2.5, color=col, lw=1, label=lab)
    ax[0].set_xlabel("SPSA iteration"); ax[0].set_ylabel("binary cross-entropy"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("SPSA iteration"); ax[1].set_ylabel("ROC-AUC"); ax[1].legend(frameon=False, loc="lower right")
    _panel(ax[0], "a", "training loss"); _panel(ax[1], "b", "training ROC-AUC")
    _save(fig, "fig_vqc.pdf")


def main():
    fig_results(); fig_calibration(); fig_vqc()


if __name__ == "__main__":
    main()
