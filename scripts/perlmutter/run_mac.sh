#!/bin/bash
#SBATCH -A m4341                  # FIXME
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_sim_sweep
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out
#
# MAC simulation sweep: per-step solver behavior across topology growth, plus
# epsilon-regularization sensitivity. Each (solver, eps) combo runs a full
# binding cascade; per-step metrics.csv is the writeup data.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1

JID=${SLURM_JOB_ID:-local}
RESULTS=results/mac_sweep_jid${JID}
mkdir -p "$RESULTS"

STEPS=${STEPS:-10000}

# Solver comparison at default eps_reg=1e-6.
for solver in pcg cg gs; do
  out="$RESULTS/solver_${solver}"
  echo "==> solver=$solver  eps_reg=1e-6"
  ./build/apps/mac_sim --steps "$STEPS" --dump_every 200 \
    --solver "$solver" --rtol 1e-8 --max_iter 2000 \
    --eps_reg 1e-6 --out_dir "$out"
done

# Epsilon-regularization sensitivity sweep w/ PCG.
for eps in 1e-4 1e-6 1e-8 1e-10; do
  out="$RESULTS/eps_${eps}"
  echo "==> solver=pcg  eps_reg=$eps"
  ./build/apps/mac_sim --steps "$STEPS" --dump_every 200 \
    --solver pcg --rtol 1e-8 --max_iter 2000 \
    --eps_reg "$eps" --out_dir "$out"
done

echo "wrote $RESULTS/{solver_*,eps_*}/metrics.csv"
