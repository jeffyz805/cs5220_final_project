#pragma once

#include <cstdint>
#include <span>

#include "la/blas.h"
#include "la/sparse_matrix.h"

namespace mac::solvers {

using la::CsrMatrix;
using la::Real;
using la::Index;

struct IterativeOpts {
    Real     rtol         = 1e-8;     // ||r||/||b|| stopping tol
    Real     atol         = 0.0;      // absolute tol on ||r|| (0 disables)
    int      max_iter     = 1000;
    bool     record_history = false;
};

struct IterativeResult {
    int      iters         = 0;
    Real     final_rresid  = 0.0;     // ||r||/||b||
    Real     final_resid   = 0.0;     // ||r||
    bool     converged     = false;
    bool     stagnated     = false;
    bool     diverged      = false;
};

// All solvers expect: A SPD (or at least positive semidefinite for tests of
// behavior under singular systems), b.size() == x.size() == A.rows() == A.cols().

IterativeResult jacobi      (const CsrMatrix& A,
                             std::span<const Real> b,
                             std::span<Real> x,
                             const IterativeOpts& opts = {});

IterativeResult gauss_seidel(const CsrMatrix& A,
                             std::span<const Real> b,
                             std::span<Real> x,
                             const IterativeOpts& opts = {});

IterativeResult rb_gauss_seidel(const CsrMatrix& A,
                                std::span<const Real> b,
                                std::span<Real> x,
                                const IterativeOpts& opts = {});

IterativeResult cg          (const CsrMatrix& A,
                             std::span<const Real> b,
                             std::span<Real> x,
                             const IterativeOpts& opts = {});

// PCG with Jacobi (diagonal) preconditioner.
IterativeResult pcg_jacobi  (const CsrMatrix& A,
                             std::span<const Real> b,
                             std::span<Real> x,
                             const IterativeOpts& opts = {});

} // namespace mac::solvers
