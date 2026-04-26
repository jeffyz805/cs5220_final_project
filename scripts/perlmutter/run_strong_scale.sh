#!/bin/bash
#SBATCH -A m4341                  # FIXME: replace w/ actual project allocation
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_strong_scale
#SBATCH -t 00:30:00
#SBATCH -N 2                      # 2 nodes max for strong scale
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out
#
# Strong scaling sweep: fixed N, vary rank count.
# Submit from project root after `cmake --build build`.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

N=${N:-1000000}                   # 10^6 by default; set N=100000 for 10^5 sweep
RESULTS=results/strong_N${N}_jid${SLURM_JOB_ID}
mkdir -p "$RESULTS"

for ranks in 1 4 16 64 128 256; do
  srun -n "$ranks" --cpu-bind=cores \
    ./build/apps/solver_bench \
      --mode synthetic --N "$N" --solver cg \
      --csv "$RESULTS/strong_r${ranks}.csv"
done
