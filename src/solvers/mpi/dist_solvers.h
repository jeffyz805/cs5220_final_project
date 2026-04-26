#pragma once

#include <span>

#include "solvers/iterative/iterative.h"
#include "solvers/mpi/distributed_csr.h"

namespace mac::solvers::mpi {

// Distributed iterative solvers. All operate on rank-local b, x spans of size
// A.n_local(). Convergence is judged on global ||r||/||b||.

IterativeResult dist_jacobi (const DistributedCsr& A,
                             std::span<const la::Real> b,
                             std::span<la::Real> x,
                             const IterativeOpts& opts = {});

IterativeResult dist_cg     (const DistributedCsr& A,
                             std::span<const la::Real> b,
                             std::span<la::Real> x,
                             const IterativeOpts& opts = {});

IterativeResult dist_pcg_jacobi(const DistributedCsr& A,
                                std::span<const la::Real> b,
                                std::span<la::Real> x,
                                const IterativeOpts& opts = {});

// Block-Jacobi outer / forward-Gauss-Seidel inner: each rank does a local
// forward-GS sweep using halo values from the previous outer iteration.
// Cross-rank coupling is Jacobi-style (degraded convergence vs true GS).
IterativeResult dist_block_gs(const DistributedCsr& A,
                              std::span<const la::Real> b,
                              std::span<la::Real> x,
                              const IterativeOpts& opts = {});

} // namespace mac::solvers::mpi
