#!/bin/bash
# P3 sanity check: MPI distributed solvers on Perlmutter.
#
# Run from project root on a compute node (interactive or via sbatch).
#   salloc -A <ALLOC> -C cpu -q interactive -t 0:30:00 -N 1
#   bash scripts/perlmutter/p3_sanity.sh
#
# Validates:
#   - Build w/ USE_MPI=ON
#   - test_mpi cross-validation: dist_cg/dist_pcg/dist_block_gs match serial at 4 ranks
#   - solver_bench dist mode runs at 1, 2, 4, 8 ranks for N in {10000, 100000}

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== module load =="
module load PrgEnv-gnu cray-mpich cmake

echo "== configure =="
export CC=cc CXX=CC
rm -rf build
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MPI=ON -DUSE_OPENMP=ON -DUSE_AVX2=ON \
  -DUSE_CHOLMOD=OFF -DUSE_EIGEN_TESTS=ON

echo "== build =="
cmake --build build -j 16

echo "== serial unit tests =="
./build/tests/mac_tests --reporter compact

echo "== MPI cross-validation (4 ranks) =="
srun -n 4 ./build/tests/test_mpi

echo "== solver_bench distributed sweep =="
mkdir -p results
OUT=results/p3_sanity.csv
rm -f "$OUT"
export OMP_NUM_THREADS=1

# Cross-rank consistency: same N, vary ranks. Only dist_cg / dist_pcg /
# dist_block_gs are expected to converge on this random BᵀB+αI input.
for solver in dist_cg dist_pcg dist_block_gs; do
  for ranks in 1 2 4 8; do
    for N in 10000 100000; do
      echo "  $solver  ranks=$ranks  N=$N"
      srun -n "$ranks" --cpu-bind=cores \
        ./build/apps/solver_bench --N "$N" --alpha 4.0 \
          --solver "$solver" --csv "$OUT" \
          --tag "p3_n${N}_r${ranks}"
    done
  done
done

# dist_jacobi: log behavior only (likely diverges on random BᵀB+αI).
for ranks in 1 4; do
  srun -n "$ranks" --cpu-bind=cores \
    ./build/apps/solver_bench --N 10000 --alpha 4.0 \
      --solver dist_jacobi --csv "$OUT" --tag "p3_jac_r${ranks}" || true
done

echo "== results =="
column -t -s, "$OUT"
echo
echo "OK — wrote $OUT"
