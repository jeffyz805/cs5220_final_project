#!/bin/bash
# P0/P1/P2 sanity check on Perlmutter.
#
# Run from project root on a compute node (interactive or via sbatch).
#   salloc -A <ALLOC> -C cpu -q interactive -t 0:30:00 -N 1
#   bash scripts/perlmutter/p012_sanity.sh
#
# Validates:
#   - CMake configure + build w/ Cray PrgEnv-gnu + cray-mpich + AVX2 (znver3)
#   - Catch2 unit tests (12 cases) all pass
#   - solver_bench runs every solver at small + medium N
#   - Outputs results/p012_sanity.csv for later analysis

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "== module load =="
module load PrgEnv-gnu cray-mpich cmake
module list 2>&1 | head -20

echo "== configure =="
export CC=cc CXX=CC
rm -rf build
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DUSE_AVX2=ON \
  -DUSE_CHOLMOD=OFF \
  -DUSE_EIGEN_TESTS=ON

echo "== build =="
cmake --build build -j 16

echo "== unit tests =="
ctest --test-dir build --output-on-failure

echo "== solver_bench sweep =="
mkdir -p results
OUT=results/p012_sanity.csv
rm -f "$OUT"
export OMP_NUM_THREADS=1   # P3 not done yet; serial only

# Small (N=300, well-cond) — every solver should be fast.
for s in cg pcg jacobi gs rbgs dense_chol; do
  ./build/apps/solver_bench --N 300 --alpha 4.0 --solver $s \
    --csv "$OUT" --tag small_alpha4
done

# Medium (N=5000, well-cond) — dense_chol still feasible (~25 MB), iter
# solvers should converge in tens-hundreds of iters.
for s in cg pcg gs rbgs; do
  ./build/apps/solver_bench --N 5000 --alpha 4.0 --solver $s \
    --csv "$OUT" --tag med_alpha4
done

# Ill-conditioned (alpha=1e-3) — record divergence/stagnation behavior.
for s in cg pcg jacobi gs rbgs; do
  ./build/apps/solver_bench --N 1000 --alpha 1e-3 --solver $s \
    --csv "$OUT" --tag illcond
done

echo "== results =="
column -t -s, "$OUT"
echo
echo "OK — wrote $OUT"
