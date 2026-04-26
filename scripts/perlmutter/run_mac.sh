#!/bin/bash
#SBATCH -A m4341                  # FIXME
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_sim
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

RANKS=${RANKS:-128}
SOLVER=${SOLVER:-pcg}
RESULTS=results/mac_${SOLVER}_r${RANKS}_jid${SLURM_JOB_ID}
mkdir -p "$RESULTS"

srun -n "$RANKS" --cpu-bind=cores \
  ./build/apps/mac_sim \
    --solver "$SOLVER" \
    --scenario data/mac_default.json \
    --steps 50000 --dump-every 100 \
    --out "$RESULTS"
