#!/bin/bash
# Generate every figure for report/report.md from the latest run results.
#
# Usage:
#   bash scripts/analysis/make_all_plots.sh                  # auto-discover
#   bash scripts/analysis/make_all_plots.sh <strong_csv> <weak_csv> <mac_dir>

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p report/figures

# Need pandas + matplotlib. Module-load on Perlmutter; harmless elsewhere.
module load python 2>/dev/null || true

PYTHON=${PYTHON:-python3}
PLOT="$PYTHON scripts/analysis/plot_scaling.py"

# Auto-discover most recent results dirs if not given.
if [ $# -eq 3 ]; then
  STRONG_CSV=$1; WEAK_CSV=$2; MAC_DIR=$3
else
  STRONG_CSV=$(ls -t results/strong_scale_jid*/strong_scale.csv 2>/dev/null | head -n1 || echo "")
  WEAK_CSV=$(  ls -t results/weak_scale_jid*/weak_scale.csv     2>/dev/null | head -n1 || echo "")
  MAC_DIR=$(   ls -td results/mac_sweep_jid*                    2>/dev/null | head -n1 || echo "")
fi

if [ -n "$STRONG_CSV" ] && [ -f "$STRONG_CSV" ]; then
  echo "==> strong scaling: $STRONG_CSV"
  $PLOT strong "$STRONG_CSV" -o report/figures/strong_scaling.png
  $PLOT comm   "$STRONG_CSV" -o report/figures/comm_breakdown.png
else
  echo "skipping strong scaling (no csv)"
fi

if [ -n "$WEAK_CSV" ] && [ -f "$WEAK_CSV" ]; then
  echo "==> weak scaling: $WEAK_CSV"
  $PLOT weak "$WEAK_CSV" -o report/figures/weak_scaling.png
else
  echo "skipping weak scaling (no csv)"
fi

if [ -n "$MAC_DIR" ] && [ -d "$MAC_DIR" ]; then
  for d in "$MAC_DIR"/solver_*/ "$MAC_DIR"/eps_*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    echo "==> mac timeseries: $name"
    $PLOT mac "$d/metrics.csv" -o "report/figures/mac_${name}.png"
  done
else
  echo "skipping mac plots (no dir)"
fi

echo
echo "figures in report/figures/"
ls report/figures 2>/dev/null || true
