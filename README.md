# MAC Constraint-Solver Engine (CS5220)

Brownian-dynamics rigid-body engine + benchmark suite of sparse linear solvers,
applied to a stylized coarse-grained simulation of complement-system MAC
(Membrane Attack Complex) assembly. Target: NERSC Perlmutter CPU partition.

Plan: see `/Users/jzhou/.claude/plans/i-ve-given-you-a-snazzy-clover.md`.

## Local build (macOS)

```sh
brew install cmake libomp open-mpi suite-sparse
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Without MPI/OpenMP installed:

```sh
cmake -S . -B build -DUSE_MPI=OFF -DUSE_OPENMP=OFF
```

## Perlmutter build

```sh
module load PrgEnv-gnu cray-mpich cmake
export CC=cc CXX=CC
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/perlmutter.cmake \
  -DUSE_CHOLMOD=ON
cmake --build build -j
```

## Run

```sh
sbatch scripts/perlmutter/run_strong_scale.sh
sbatch scripts/perlmutter/run_weak_scale.sh
sbatch scripts/perlmutter/run_mac.sh
```

## Layout

- `src/la/`       — sparse matrix (CSR), AVX2 vector kernels
- `src/solvers/`  — direct (dense Cholesky, CHOLMOD) + iterative (Jacobi, GS, RBGS, CG, PCG) + MPI variants
- `src/physics/`  — rigid bodies, overdamped Langevin integrator
- `src/constraints/` — distance / ball-joint / weld + Jacobian assembler
- `src/spatial/`  — cell list for proximity
- `src/mac/`      — protein definitions, binding cascade
- `bench/`        — synthetic SPD benchmark
- `apps/`         — `solver_bench`, `mac_sim`
- `tests/`        — Catch2 unit + integration
- `scripts/perlmutter/` — SLURM batch
- `scripts/analysis/`   — Python plot generators
