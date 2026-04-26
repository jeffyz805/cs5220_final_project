#include "solvers/mpi/dist_solvers.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "la/blas.h"
#include "solvers/mpi/distributed_blas.h"

namespace mac::solvers::mpi {

using la::Index;
using la::Real;

namespace {

bool finite(Real v) { return std::isfinite(v); }

bool stop(IterativeResult& res, Real rnorm, Real bnorm,
          const IterativeOpts& opts, int it) {
    res.iters        = it;
    res.final_resid  = rnorm;
    res.final_rresid = (bnorm > 0.0) ? rnorm / bnorm : rnorm;
    if (!finite(rnorm)) { res.diverged = true; return true; }
    if (rnorm <= opts.atol) { res.converged = true; return true; }
    if (bnorm > 0.0 && rnorm / bnorm <= opts.rtol) { res.converged = true; return true; }
    return false;
}

// Compute r = b - A x (distributed).
void dist_residual(const DistributedCsr& A,
                   std::span<const Real> b,
                   std::span<const Real> x,
                   std::span<Real> r) {
    A.spmv(x, r);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
}

} // anon

// ---------------- distributed Jacobi ----------------
//
// x_new = x + D^-1 (b - A x). Single buffer; no x_old needed.

IterativeResult dist_jacobi(const DistributedCsr& A,
                            std::span<const Real> b,
                            std::span<Real> x,
                            const IterativeOpts& opts) {
    const Index n = A.n_local();
    assert(static_cast<Index>(b.size()) == n);
    assert(static_cast<Index>(x.size()) == n);

    std::vector<Real> diag_inv(n);
    for (Index i = 0; i < n; ++i) {
        Real d = A.diag_local(i);
        assert(d != 0.0 && "distributed Jacobi: zero diagonal");
        diag_inv[i] = 1.0 / d;
    }

    std::vector<Real> r(n);
    Real bnorm = dist_nrm2(A.comm(), b);

    dist_residual(A, b, x, r);
    Real rnorm = dist_nrm2(A.comm(), r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    for (int it = 1; it <= opts.max_iter; ++it) {
        for (Index i = 0; i < n; ++i) x[i] += diag_inv[i] * r[i];
        dist_residual(A, b, x, r);
        rnorm = dist_nrm2(A.comm(), r);
        if (stop(res, rnorm, bnorm, opts, it)) return res;
    }
    res.stagnated = true;
    return res;
}

// ---------------- distributed CG ----------------

IterativeResult dist_cg(const DistributedCsr& A,
                        std::span<const Real> b,
                        std::span<Real> x,
                        const IterativeOpts& opts) {
    const Index n = A.n_local();
    assert(static_cast<Index>(b.size()) == n);
    assert(static_cast<Index>(x.size()) == n);

    std::vector<Real> r(n), p(n), Ap(n);
    Real bnorm = dist_nrm2(A.comm(), b);

    dist_residual(A, b, x, r);
    Real rnorm = dist_nrm2(A.comm(), r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    la::copy(r, p);
    Real rTr = dist_dot(A.comm(), r, r);

    for (int it = 1; it <= opts.max_iter; ++it) {
        A.spmv(p, Ap);
        Real pAp = dist_dot(A.comm(), p, Ap);
        if (!finite(pAp) || pAp <= 0.0) {
            res.iters     = it;
            res.diverged  = !finite(pAp);
            res.stagnated = !res.diverged;
            return res;
        }
        Real alpha = rTr / pAp;
        la::axpy( alpha, p,  x);
        la::axpy(-alpha, Ap, r);

        Real rTr_new = dist_dot(A.comm(), r, r);
        rnorm = std::sqrt(rTr_new);
        if (stop(res, rnorm, bnorm, opts, it)) return res;

        Real beta = rTr_new / rTr;
        rTr = rTr_new;
        la::xpay(beta, r, p);  // p = r + beta * p
    }
    res.stagnated = true;
    return res;
}

// ---------------- distributed PCG (Jacobi precond) ----------------

IterativeResult dist_pcg_jacobi(const DistributedCsr& A,
                                std::span<const Real> b,
                                std::span<Real> x,
                                const IterativeOpts& opts) {
    const Index n = A.n_local();
    std::vector<Real> diag_inv(n);
    for (Index i = 0; i < n; ++i) {
        Real d = A.diag_local(i);
        assert(d > 0.0);
        diag_inv[i] = 1.0 / d;
    }

    std::vector<Real> r(n), z(n), p(n), Ap(n);
    Real bnorm = dist_nrm2(A.comm(), b);

    dist_residual(A, b, x, r);
    Real rnorm = dist_nrm2(A.comm(), r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    for (Index i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];
    la::copy(z, p);
    Real rTz = dist_dot(A.comm(), r, z);

    for (int it = 1; it <= opts.max_iter; ++it) {
        A.spmv(p, Ap);
        Real pAp = dist_dot(A.comm(), p, Ap);
        if (!finite(pAp) || pAp <= 0.0) {
            res.iters     = it;
            res.diverged  = !finite(pAp);
            res.stagnated = !res.diverged;
            return res;
        }
        Real alpha = rTz / pAp;
        la::axpy( alpha, p,  x);
        la::axpy(-alpha, Ap, r);

        rnorm = dist_nrm2(A.comm(), r);
        if (stop(res, rnorm, bnorm, opts, it)) return res;

        for (Index i = 0; i < n; ++i) z[i] = diag_inv[i] * r[i];
        Real rTz_new = dist_dot(A.comm(), r, z);
        Real beta = rTz_new / rTz;
        rTz = rTz_new;
        la::xpay(beta, z, p);
    }
    res.stagnated = true;
    return res;
}

// ---------------- distributed block-Jacobi outer / forward-GS inner ----------------
//
// Halo values are frozen at the start of each outer iteration. Within rank we
// run a forward Gauss-Seidel sweep: for each row i, owned cols j < i use the
// already-updated x[j], and halo cols pull from the frozen halo buffer.

IterativeResult dist_block_gs(const DistributedCsr& A,
                              std::span<const Real> b,
                              std::span<Real> x,
                              const IterativeOpts& opts) {
    const Index n = A.n_local();
    Real bnorm = dist_nrm2(A.comm(), b);

    std::vector<Real> r(n);
    dist_residual(A, b, x, r);
    Real rnorm = dist_nrm2(A.comm(), r);

    IterativeResult res;
    if (stop(res, rnorm, bnorm, opts, 0)) return res;

    auto rp = A.local().row_ptr();
    auto ci = A.local().col_ind();
    auto vs = A.local().values();

    for (int it = 1; it <= opts.max_iter; ++it) {
        // Freeze halo from current x.
        A.fetch_halo(x);
        auto halo = A.halo_buf();

        // Local forward-GS sweep. owned cols indexed in [0, n_local), halo
        // cols in [n_local, n_local + n_halo).
        for (Index i = 0; i < n; ++i) {
            Real sigma = 0.0;
            Real diag  = 0.0;
            for (Index k = rp[i]; k < rp[i + 1]; ++k) {
                Index c = ci[k];
                if (c == i) { diag = vs[k]; continue; }
                Real xv = (c < n) ? x[c] : halo[c - n];
                sigma += vs[k] * xv;
            }
            assert(diag != 0.0);
            x[i] = (b[i] - sigma) / diag;
        }

        dist_residual(A, b, x, r);
        rnorm = dist_nrm2(A.comm(), r);
        if (stop(res, rnorm, bnorm, opts, it)) return res;
    }
    res.stagnated = true;
    return res;
}

} // namespace mac::solvers::mpi
