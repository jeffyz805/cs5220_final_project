#!/bin/bash
#SBATCH -A m4341                  # FIXME: replace w/ actual project allocation
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_weak_scale
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out
#
# Weak scaling: N grows proportionally with ranks.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

N_PER_RANK=${N_PER_RANK:-10000}
RESULTS=results/weak_npr${N_PER_RANK}_jid${SLURM_JOB_ID}
mkdir -p "$RESULTS"

for ranks in 1 4 16 64 128 256; do
  N=$((N_PER_RANK * ranks))
  srun -n "$ranks" --cpu-bind=cores \
    ./build/apps/solver_bench \
      --mode synthetic --N "$N" --solver cg \
      --csv "$RESULTS/weak_r${ranks}.csv"
done
