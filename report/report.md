# Sparse Linear Solvers for a Stylized Complement-Cascade Brownian-Dynamics Engine

CS 5220 Final Project — Andrew J. Kim, Jefferson Zhou, Jedi Lertviwatkul

## Abstract

TODO: 150-200 words. Restate hypothesis, summarize headline findings (Krylov
robustness vs stationary-method fragility under partitioning; iter count tracks
constraint count in MAC; strong-scaling efficiency at multi-node; comm-vs-compute
crossover).

## 1. Background

### 1.1 Constraint-based overdamped Langevin dynamics

TODO: equations from §P4 of the plan. Per-step solve of `(J M^-1 J^T + εI) λ =
-β C - J M^-1 F`, integrate `x ← x + (Δt/γ) F_total`. Equality-only constraints
(distance / ball-joint / weld). Baumgarte feedback. Cite proposal §1, §4.

### 1.2 Coarse-grained MAC assembly as a workload

TODO: explicit caveat — this is a *stylized* coarse-grained model inspired by
the complement terminal pathway, not biologically validated. Each protein =
single rigid body with named binding sites. Welds inserted dynamically at
runtime to drive growing constraint matrix.

## 2. Implementation

### 2.1 Linear-algebra core

- CSR sparse matrix with incremental row add/remove builder.
- AVX2-vectorized vector primitives (axpy, dot, nrm2, scal); scalar fallback
  on non-x86 dev hardware.
- Row-parallel SpMV with optional OpenMP.

### 2.2 Solver suite

| Solver | Serial | OpenMP | MPI |
|---|---|---|---|
| Dense Cholesky (from-scratch) | ✓ | — | — |
| Jacobi | ✓ | ✓ | ✓ |
| Forward Gauss-Seidel | ✓ | — | — |
| Red-black Gauss-Seidel | ✓ | ✓ | ✓ |
| Block-Jacobi outer / GS inner | — | — | ✓ |
| Conjugate Gradient | ✓ | ✓ | ✓ |
| PCG (Jacobi precond) | ✓ | ✓ | ✓ |

Dropped: from-scratch sparse Cholesky (symbolic factorization out of solo
scope); CHOLMOD wrapper (deferred, not used in this report).

### 2.3 Distributed primitives

- 1D row-block partition.
- Halo-exchange SpMV: column renumbering at construction (owned in
  `[0, n_local)`, halo in `[n_local, n_local + n_halo)`); per-neighbor
  send/recv lists built once via Alltoall + Alltoallv, reused every SpMV.
- Distributed dot/nrm2 via Allreduce.
- All comms timed via internal counters (Alltoallv, Allreduce, local SpMV,
  pack) to enable comm-vs-compute breakdown.

### 2.4 Physics engine

TODO: tiny linear-algebra (Vec3, Quat, Mat3); RigidBody (isotropic mobility);
overdamped Langevin step; constraint Jacobian assembler emitting CSR.

### 2.5 MAC scenario

TODO: protein cascade specs (C5b → C6 → C7 → C8 → C9, with C9 homotypic
chain), stochastic binding rule (proximity gate; weld inserted with
rest_orient set so q_err = 1 at bind time), confining harmonic well to keep
bodies bounded.

## 3. Verification

- 21 Catch2 unit/integration cases pass on Perlmutter
  (`build/tests/mac_tests`). Coverage: SpMV vs Eigen oracle, every iterative
  solver on well-conditioned SPD, dense Cholesky residual ≤ 1e-12,
  conditioning sweep, free Brownian MSD = 6Dt within 8%, distance-constraint
  Baumgarte relaxation, scenario monotonicity.
- MPI cross-validation: each distributed solver matches its serial
  counterpart within rel-err 1e-6 across 1/2/4/8 ranks
  (`build/tests/test_mpi`).

## 4. Synthetic SPD Benchmark

Workload: `A = BᵀB + αI` with `B` sparse Gaussian (`nnz_per_row=8`), α
controls conditioning.

### 4.1 Single-node correctness

TODO: include `results/p3_sanity.csv` summary table (already collected in
P3).

Headline finding from P3 sanity:

- `dist_cg`/`dist_pcg`: converge in 30-34 iters at all rank counts (1/2/4/8)
  for N ∈ {10⁵, 10⁶}. Strong scaling already visible at single-node SHM —
  PCG at N=10⁵ goes from 0.18s @ 1 rank to 0.035s @ 8 ranks (≈5×).
- `dist_block_gs`: converges at 1-2 ranks, **diverges to inf at 4+ ranks**.
  Block-Jacobi outer / GS inner has no convergence guarantee unless A is
  block-diagonally-dominant; random `BᵀB+αI` is not. As ranks ↑, partition
  splits coupling across blocks and Jacobi-style outer overwhelms GS
  smoothing.
- `dist_jacobi`: diverges at all ranks (intrinsic — same diagonal-dominance
  issue).

> **Research finding:** stationary methods are fragile under
> domain partitioning even when SPD is preserved. Krylov methods are robust.

### 4.2 Strong scaling (multi-node)

TODO: insert plot `report/figures/strong_scaling.png` produced by:
```
python scripts/analysis/plot_scaling.py strong \
  results/strong_scale_jid<JID>/strong_scale.csv \
  -o report/figures/strong_scaling.png
```

Comments: speedup vs ideal, parallel efficiency drop-off, where it saturates.

### 4.3 Weak scaling (multi-node)

TODO: insert plot `report/figures/weak_scaling.png`.

Comments: per-rank time vs rank count; ideal is flat. Where the curve climbs
indicates where comm overhead starts dominating.

### 4.4 Comm-vs-compute breakdown

TODO: insert `report/figures/comm_breakdown.png`.

Decompose solve time into local SpMV / Alltoallv (halo) / Allreduce (dot) /
pack overhead. Identify the rank count at which comm overtakes compute for
each N.

## 5. MAC Binding-Event Workload

### 5.1 Per-step time series

TODO: insert `report/figures/mac_pcg_timeseries.png` from
`results/mac_sweep_*/solver_pcg/metrics.csv`.

Show: constraint count, solver iters, residual, and per-step solve time over
the cascade. Annotate binding events.

### 5.2 Solver comparison on MAC workload

From P5 sanity (3000 steps, 25 bodies, ~18 binding events at end):

| Solver | iters @ cons=18 | wall (3000 steps) | Final rresid |
|---|---|---|---|
| PCG | 124 | 0.51s | 1e-9 ✓ |
| CG  | 128 | 0.50s | 1e-9 ✓ |
| GS  | **1000 (capped)** | 3.91s | ~1e-7 ✗ |

**GS stagnates above ~16-18 weld constraints**, hits max_iter cap. ~8×
wall-time penalty. Baumgarte stabilization absorbs the residual sloppiness so
the simulation remains physically stable, but constraint accuracy is
degraded.

### 5.3 Epsilon-regularization sensitivity

TODO: from `results/mac_sweep_*/eps_*/metrics.csv` show how iter count and
final rresid trade against ε ∈ {1e-4, 1e-6, 1e-8, 1e-10}. Larger ε = better
conditioned but more constraint slop.

## 6. Discussion

TODO:

- Connect synthetic-SPD findings (block-GS divergence, CG/PCG robustness)
  to the MAC results (GS stagnation under welded chains).
- Comment on whether the MAC binding-event "stiffness spike" the proposal
  hypothesized actually appears (look at iter-jump at binding events in
  §5.1 plot).
- Limitations: stylized model, single-rigid-body proteins, no contact, no
  inter-node MAC simulation (mac_sim is serial — solver scaling story is
  separate).

## 7. Conclusion

TODO: 100-150 words. Direct support for the proposal hypothesis. Concrete
guidance: prefer Krylov for dynamic-topology constraint solvers; reserve
stationary methods for diagonally-dominant systems or when used as inner
smoothers within block-Jacobi.

## Acknowledgments

NERSC Perlmutter (allocation: TODO). HPC research group at Cornell.

## References

TODO: MUMPS paper, Ihm et al. 2004 SIMD physics, O'Leary 1990 Krylov survey,
Baraff-Witkin SIGGRAPH course on rigid-body physics, complement-cascade
biology references.
