#!/usr/bin/env bash
# QBioBench end-to-end driver. Stages run in order; each can be called alone.
# No stage below spends QPU time. Hardware execution is documented in README.md.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
PY="uv run python"
JOBS="${JOBS:-16}"
# Optional local extension, not part of the public release: a reproduce.local.sh next to this script may define
# replay_local and analysis_local, which the replay and analysis stages call when they are defined.
if [[ -f reproduce.local.sh ]]; then source reproduce.local.sh; fi
has() { declare -F "$1" > /dev/null; }

features() {
  for t in apnea ptbxl wesad; do $PY scripts/build_features.py --task "$t" --subset capped; done
  for t in apnea ptbxl; do $PY scripts/build_features.py --task "$t" --subset full; done
}

baselines() {
  $PY experiments/run_baselines.py --task all --subset capped
  $PY experiments/run_baselines.py --task apnea --subset full
  $PY experiments/run_baselines.py --task ptbxl --subset full
}

screen() {
  $PY experiments/screen.py --task all --mode noisy --jobs "$JOBS" --tag screen_v2
  $PY experiments/screen.py --task all --mode sv --jobs "$JOBS" --tag screen_v2
  $PY experiments/screen.py --task all --mode sv --models vqc --vqc-loss one-sided --jobs "$JOBS" --tag screen_v2
}

replay() {
  if has replay_local; then replay_local; fi
  $PY experiments/hw_replay2.py --tag phoenix_2026-09
}

analysis() {
  # sequential on purpose: concurrent runs oversubscribe XGBoost threads
  $PY experiments/analysis_hw.py --tag phoenix_2026-09 --scope pool
  if has analysis_local; then analysis_local; fi
  $PY experiments/describe_data.py
}

tables() {
  # summary tables (tables/*.csv) and data figures (figures/*.pdf), generated locally from data/results/ only
  $PY experiments/make_tables.py
  $PY experiments/figures_v2.py
}

paper() {
  # tables and figures, then, only in a tree that holds the paper sources (manuscript/ and its generators):
  # in-text numbers (fails on an undefined key) -> house-style and cross-reference checks -> PDFs of the paper and
  # the supplementary information (they cite each other through xr, so each is rebuilt once the other's .aux exists)
  # -> scan of the typeset text for identifying strings
  tables
  if [[ -d manuscript && -f experiments/make_numbers.py ]]; then
    $PY experiments/make_numbers.py --check
    $PY experiments/check_manuscript.py
    (cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex && latexmk -pdf -interaction=nonstopmode supplement.tex \
      && latexmk -g -pdf -interaction=nonstopmode main.tex && latexmk -g -pdf -interaction=nonstopmode supplement.tex)
    $PY scripts/check_anonymity.py --text-only manuscript/main.pdf manuscript/supplement.pdf
  else
    echo "paper: no manuscript/ in this tree, tables and figures only"
  fi
}

all() { features; baselines; screen; replay; analysis; tables; }

"${1:-all}"
