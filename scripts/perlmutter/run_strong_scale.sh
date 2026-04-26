#!/bin/bash
#SBATCH -A m4341                  # FIXME: replace w/ actual project allocation
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_strong_scale
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out
#
# Strong scaling sweep: fixed N, vary rank count from 1 to 256 (2 nodes).
# Sweeps multiple solvers + two N (10^5, 10^6).
#
# Submit from project root after `cmake --build build -j`.
# Outputs: results/strong_scale_jid${JOBID}/strong_scale.csv

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

JID=${SLURM_JOB_ID:-local}
RESULTS=results/strong_scale_jid${JID}
mkdir -p "$RESULTS"
OUT=$RESULTS/strong_scale.csv

SOLVERS=(dist_cg dist_pcg)
RANK_COUNTS=(1 4 16 64 128 256)
NS=(100000 1000000)

for solver in "${SOLVERS[@]}"; do
  for N in "${NS[@]}"; do
    for ranks in "${RANK_COUNTS[@]}"; do
      echo "==> solver=$solver  N=$N  ranks=$ranks"
      srun -n "$ranks" --cpu-bind=cores \
        ./build/apps/solver_bench --N "$N" --alpha 4.0 \
          --solver "$solver" --csv "$OUT" \
          --tag "strong_${solver}_N${N}_r${ranks}"
    done
  done
done

echo "wrote $OUT"
