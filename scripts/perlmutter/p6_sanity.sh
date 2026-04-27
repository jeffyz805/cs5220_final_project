#!/bin/bash
# P6 sanity check: light-touch scaling + plot generation, single node.
#
#   salloc -A <ALLOC> -C cpu -q interactive -t 0:30:00 -N 1
#   bash scripts/perlmutter/p6_sanity.sh
#
# Validates:
#   - solver_bench MPI mode emits new comm-vs-compute columns
#   - plot generators produce all 4 figure types
# Cheap (single node, small N) — full multi-node sweeps come from
# run_strong_scale.sh + run_weak_scale.sh + run_mac.sh as separate sbatch jobs.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ ! -f build/CMakeCache.txt ]; then
  module load PrgEnv-gnu cray-mpich cmake
  export CC=cc CXX=CC
  cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPI=ON -DUSE_OPENMP=ON -DUSE_AVX2=ON
fi

# Plotting needs a real Python with pandas + matplotlib.
module load python 2>/dev/null || true

echo "== build =="
cmake --build build -j 16

mkdir -p results
JID=local_p6
STRONG=results/strong_p6_${JID}.csv
WEAK=results/weak_p6_${JID}.csv
rm -f "$STRONG" "$WEAK"

export OMP_NUM_THREADS=1

echo "== mini strong sweep (single node) =="
for solver in dist_cg dist_pcg; do
  for ranks in 1 2 4 8; do
    srun -n "$ranks" --cpu-bind=cores \
      ./build/apps/solver_bench --N 100000 --alpha 4.0 \
        --solver "$solver" --csv "$STRONG" \
        --tag "p6_strong_${solver}_r${ranks}"
  done
done

echo "== mini weak sweep (single node) =="
for solver in dist_cg dist_pcg; do
  for ranks in 1 2 4 8; do
    N=$((10000 * ranks))
    srun -n "$ranks" --cpu-bind=cores \
      ./build/apps/solver_bench --N "$N" --alpha 4.0 \
        --solver "$solver" --csv "$WEAK" \
        --tag "p6_weak_${solver}_r${ranks}"
  done
done

echo "== mac smoke run =="
mkdir -p results/mac_p6_smoke
./build/apps/mac_sim --steps 2000 --dump_every 100 \
  --solver pcg --out_dir results/mac_p6_smoke

echo "== plots =="
PYTHON=${PYTHON:-python3}
mkdir -p report/figures
$PYTHON scripts/analysis/plot_scaling.py strong "$STRONG" \
  -o report/figures/strong_p6_smoke.png
$PYTHON scripts/analysis/plot_scaling.py weak   "$WEAK"   \
  -o report/figures/weak_p6_smoke.png
$PYTHON scripts/analysis/plot_scaling.py comm   "$STRONG" \
  -o report/figures/comm_p6_smoke.png
$PYTHON scripts/analysis/plot_scaling.py mac    results/mac_p6_smoke/metrics.csv \
  -o report/figures/mac_p6_smoke.png

ls -lh report/figures/*p6_smoke.png
echo "OK"
