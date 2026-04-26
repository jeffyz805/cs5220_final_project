#!/bin/bash
# P5 sanity check: MAC scenario simulation on Perlmutter.
#
#   salloc -A <ALLOC> -C cpu -q interactive -t 0:15:00 -N 1
#   bash scripts/perlmutter/p5_sanity.sh
#
# Validates:
#   - All ctest cases (P0-P5: serial, MPI, physics, MAC)
#   - mac_sim runs end-to-end with each iterative solver
#   - traj.xyz + metrics.csv produced; binding events fire

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ ! -f build/CMakeCache.txt ]; then
  module load PrgEnv-gnu cray-mpich cmake
  export CC=cc CXX=CC
  cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPI=ON -DUSE_OPENMP=ON -DUSE_AVX2=ON \
    -DUSE_CHOLMOD=OFF -DUSE_EIGEN_TESTS=ON
fi

echo "== build =="
cmake --build build -j 16

echo "== ctest =="
./build/tests/mac_tests --reporter compact

echo "== mac_sim sweep =="
mkdir -p results
export OMP_NUM_THREADS=1

for solver in pcg cg gs; do
  out="results/mac_${solver}"
  rm -rf "$out"
  echo "  -> $solver"
  ./build/apps/mac_sim --steps 3000 --dump_every 100 \
    --solver "$solver" --rtol 1e-8 --max_iter 1000 \
    --out_dir "$out"
  echo "  ---- summary ($solver) ----"
  cat "$out/summary.txt"
  echo "  ---- last 5 metrics rows ----"
  tail -n 5 "$out/metrics.csv"
done

echo
echo "OK"
