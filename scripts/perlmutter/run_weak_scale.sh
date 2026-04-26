#!/bin/bash
#SBATCH -A m4341                  # FIXME
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J mac_weak_scale
#SBATCH -t 00:45:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -o slurm-%x-%j.out
#
# Weak scaling: N grows linearly with ranks. N_per_rank fixed.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load PrgEnv-gnu cray-mpich
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

JID=${SLURM_JOB_ID:-local}
N_PER_RANK=${N_PER_RANK:-10000}
RESULTS=results/weak_scale_jid${JID}_npr${N_PER_RANK}
mkdir -p "$RESULTS"
OUT=$RESULTS/weak_scale.csv

SOLVERS=(dist_cg dist_pcg)
RANK_COUNTS=(1 4 16 64 128 256)

for solver in "${SOLVERS[@]}"; do
  for ranks in "${RANK_COUNTS[@]}"; do
    N=$((N_PER_RANK * ranks))
    echo "==> solver=$solver  N=$N (=$N_PER_RANK x $ranks)  ranks=$ranks"
    srun -n "$ranks" --cpu-bind=cores \
      ./build/apps/solver_bench --N "$N" --alpha 4.0 \
        --solver "$solver" --csv "$OUT" \
        --tag "weak_${solver}_npr${N_PER_RANK}_r${ranks}"
  done
done

echo "wrote $OUT"
