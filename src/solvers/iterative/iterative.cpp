#include "solvers/iterative/iterative.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace mac::solvers {

namespace {

bool finite(Real v) { return std::isfinite(v); }

// Common header for iterative loops: compute initial r = b - Ax and ||b||,
// returns a non-converged result with that initial residual recorded.
struct Init { std::vector<Real> r; Real bnorm; Real rnorm0; };

Init init_residual(const CsrMatrix& A,
                   std::span<const Real> b,
                   std::span<Real> x) {
    Init it{};
    it.r.assign(b.size(), 0.0);
    la::residual(A, b, x, it.r);
    it.bnorm  = la::nrm2(b);
    it.rnorm0 = la::nrm2(it.r);
    return it;
}

bool stop(IterativeResult& res, Real rnorm, Real bnorm, const IterativeOpts& opts, int it) {
    res.iters        = it;
    res.final_resid  = rnorm;
    res.final_rresid = (bnorm > 0.0) ? rnorm / bnorm : rnorm;
    if (!finite(rnorm)) { res.diverged = true; return true; }
    if (rnorm <= opts.atol) { res.converged = true; return true; }
    if (bnorm > 0.0 && rnorm / bnorm <= opts.rtol) { res.converged = true; return true; }
    return false;
}

} // anon

// ---------------- Jacobi ----------------
//
// x_new[i] = (b[i] - sum_{j != i} A[i,j] * x_old[j]) / A[i,i]
// One full sweep per "iteration"; uses two buffers (old/new).

IterativeResult jacobi(const CsrMatrix& A,
                       std::span<const Real> b,
                       std::span<Real> x,
                       const IterativeOpts& opts) {
    assert(A.rows() == A.cols());
    const Index n = A.rows();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vs = A.values();

    std::vector<Real> diag(n);
    for (Index i = 0; i < n; ++i) {
        diag[i] = A.diag(i);
        assert(diag[i] != 0.0 && "Jacobi requires nonzero diagonal");
    }

    std::vector<Real> xnew(n);
    auto init = init_residual(A, b, x);
    IterativeResult res;
    if (stop(res, init.rnorm0, init.bnorm, opts, 0)) return res;

    for (int it = 1; it <= opts.max_iter; ++it) {
        for (Index i = 0; i < n; ++i) {
            Real sigma = 0.0;
            for (Index k = rp[i]; k < rp[i + 1]; ++k) {
                Index j = ci[k];
                if (j != i) sigma += vs[k] * x[j];
            }
            xnew[i] = (b[i] - sigma) / diag[i];
        }
        std::copy(xnew.begin(), xnew.end(), x.begin());

        std::vector<Real> r(n);
        la::residual(A, b, x, r);
        Real rn = la::nrm2(r);
        if (stop(res, rn, init.bnorm, opts, it)) return res;
    }
    res.stagnated = true;
    return res;
}

// ---------------- Forward Gauss-Seidel ----------------
// In-place sweep: for i = 0..n-1, use updated x[j] for j < i.

IterativeResult gauss_seidel(const CsrMatrix& A,
                             std::span<const Real> b,
                             std::span<Real> x,
                             const IterativeOpts& opts) {
    assert(A.rows() == A.cols());
    const Index n = A.rows();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vs = A.values();

    auto init = init_residual(A, b, x);
    IterativeResult res;
    if (stop(res, init.rnorm0, init.bnorm, opts, 0)) return res;

    for (int it = 1; it <= opts.max_iter; ++it) {
        for (Index i = 0; i < n; ++i) {
            Real sigma = 0.0;
            Real diag = 0.0;
            for (Index k = rp[i]; k < rp[i + 1]; ++k) {
                Index j = ci[k];
                if (j == i) diag = vs[k];
                else        sigma += vs[k] * x[j];
            }
            assert(diag != 0.0);
            x[i] = (b[i] - sigma) / diag;
        }
        std::vector<Real> r(n);
        la::residual(A, b, x, r);
        Real rn = la::nrm2(r);
        if (stop(res, rn, init.bnorm, opts, it)) return res;
    }
    res.stagnated = true;
    return res;
}

// ---------------- Red-Black Gauss-Seidel ----------------
//
// Color rows by parity of index for now (works perfectly on PDE-like 1D/2D
// regular grids, and is a reasonable stand-in for parallel GS on randomly
// structured matrices — which is the worst case where it loses convergence
// rate vs forward GS but trivially parallelizes).
//
// For real PDE adjacency, a true graph 2-coloring would be the right thing.
// We can swap that in later if needed.

IterativeResult rb_gauss_seidel(const CsrMatrix& A,
                                std::span<const Real> b,
                                std::span<Real> x,
                                const IterativeOpts& opts) {
    assert(A.rows() == A.cols());
    const Index n = A.rows();
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vs = A.values();

    auto init = init_residual(A, b, x);
    IterativeResult res;
    if (stop(res, init.rnorm0, init.bnorm, opts, 0)) return res;

    auto sweep = [&](int color) {
#if MAC_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (Index i = color; i < n; i += 2) {
            Real sigma = 0.0;
            Real diag = 0.0;
            for (Index k = rp[i]; k < rp[i + 1]; ++k) {
                Index j = ci[k];
                if (j == i) diag = vs[k];
                else        sigma += vs[k] * x[j];
            }
            x[i] = (b[i] - sigma) / diag;
        }
    };

    for (int it = 1; it <= opts.max_iter; ++it) {
        sweep(0);
        sweep(1);
        std::vector<Real> r(n);
        la::residual(A, b, x, r);
        Real rn = la::nrm2(r);
        if (stop(res, rn, init.bnorm, opts, it)) return res;
    }
    res.stagnated = true;
    return res;
}

// ---------------- Conjugate Gradient ----------------

IterativeResult cg(const CsrMatrix& A,
                   std::span<const Real> b,
                   std::span<Real> x,
                   const IterativeOpts& opts) {
    assert(A.rows() == A.cols());
    const std::size_t n = A.rows();

    std::vector<Real> r(n), p(n), Ap(n);
    la::residual(A, b, x, r);
    Real bnorm = la::nrm2(b);
    Real rnorm = la::nrm2(r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    la::copy(r, p);
    Real rTr = la::dot(r, r);

    for (int it = 1; it <= opts.max_iter; ++it) {
        la::spmv(A, p, Ap);
        Real pAp = la::dot(p, Ap);
        if (!finite(pAp) || pAp <= 0.0) {
            res.iters    = it;
            res.diverged = !finite(pAp);
            res.stagnated = !res.diverged;
            return res;
        }
        Real alpha = rTr / pAp;
        la::axpy( alpha, p,  x);
        la::axpy(-alpha, Ap, r);

        Real rTr_new = la::dot(r, r);
        rnorm = std::sqrt(rTr_new);
        if (stop(res, rnorm, bnorm, opts, it)) return res;

        Real beta = rTr_new / rTr;
        rTr = rTr_new;
        // p = r + beta * p
        la::xpay(beta, r, p);
    }
    res.stagnated = true;
    return res;
}

// ---------------- Preconditioned CG (Jacobi precond) ----------------

IterativeResult pcg_jacobi(const CsrMatrix& A,
                           std::span<const Real> b,
                           std::span<Real> x,
                           const IterativeOpts& opts) {
    assert(A.rows() == A.cols());
    const Index n = A.rows();

    std::vector<Real> diag_inv(n);
    for (Index i = 0; i < n; ++i) {
        Real d = A.diag(i);
        assert(d > 0.0 && "PCG-Jacobi requires positive diagonal");
        diag_inv[i] = 1.0 / d;
    }

    std::vector<Real> r(n), z(n), p(n), Ap(n);
    la::residual(A, b, x, r);
    Real bnorm = la::nrm2(b);
    Real rnorm = la::nrm2(r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    for (Index i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];
    la::copy(z, p);
    Real rTz = la::dot(r, z);

    for (int it = 1; it <= opts.max_iter; ++it) {
        la::spmv(A, p, Ap);
        Real pAp = la::dot(p, Ap);
        if (!finite(pAp) || pAp <= 0.0) {
            res.iters    = it;
            res.diverged = !finite(pAp);
            res.stagnated = !res.diverged;
            return res;
        }
        Real alpha = rTz / pAp;
        la::axpy( alpha, p,  x);
        la::axpy(-alpha, Ap, r);

        rnorm = la::nrm2(r);
        if (stop(res, rnorm, bnorm, opts, it)) return res;

        for (Index i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];
        Real rTz_new = la::dot(r, z);
        Real beta = rTz_new / rTz;
        rTz = rTz_new;
        la::xpay(beta, z, p);  // p = z + beta * p
    }
    res.stagnated = true;
    return res;
}

} // namespace mac::solvers
