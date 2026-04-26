// Solver unit tests vs Eigen oracle.
//
// Generation: A = B^T B + alpha I, where B is a sparse-Gaussian matrix.
// alpha controls conditioning. With B drawn random, A is SPD.
//
// Pass/fail thresholds intentionally mirror the plan's validation goals:
//   - Well-conditioned (kappa <~ 1e3): every iter solver hits rtol = 1e-8.
//   - Direct (dense Cholesky): residual <= 1e-12.
//   - vs Eigen oracle: relative error <= 1e-6.
//
// Ill-conditioned tests record behavior; they're tagged [conditioning] and
// don't enforce convergence.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "la/blas.h"
#include "la/sparse_matrix.h"
#include "solvers/direct/dense_cholesky.h"
#include "solvers/iterative/iterative.h"

using mac::la::CsrMatrix;
using mac::la::CsrBuilder;
using mac::la::Index;
using mac::la::Real;

namespace {

// Generate sparse SPD A = B^T B + alpha*I in CSR form, plus a dense Eigen copy
// of A for oracle solves. nnz_per_row is the average non-zero count of B.
struct SpdSystem {
    CsrMatrix A;
    Eigen::MatrixXd Adense;
    std::vector<Real> b;
    std::vector<Real> x_true;
};

SpdSystem make_spd(Index n, int nnz_per_row, Real alpha, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<Index> col_dist(0, n - 1);
    std::normal_distribution<Real> val_dist(0.0, 1.0);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, n);
    for (Index r = 0; r < n; ++r) {
        for (int k = 0; k < nnz_per_row; ++k) {
            Index c = col_dist(rng);
            B(r, c) += val_dist(rng);
        }
    }
    Eigen::MatrixXd A = B.transpose() * B
                      + alpha * Eigen::MatrixXd::Identity(n, n);

    // Build CSR from the dense A (test sizes are small).
    CsrBuilder bld(n, n);
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            Real v = A(i, j);
            if (v != 0.0) bld.push(i, j, v);
        }
    }

    SpdSystem sys;
    sys.A = bld.finalize();
    sys.Adense = A;

    sys.x_true.resize(n);
    for (auto& xi : sys.x_true) xi = val_dist(rng);

    Eigen::VectorXd xv(n);
    for (Index i = 0; i < n; ++i) xv[i] = sys.x_true[i];
    Eigen::VectorXd bv = A * xv;
    sys.b.resize(n);
    for (Index i = 0; i < n; ++i) sys.b[i] = bv[i];
    return sys;
}

Real rel_resid(const CsrMatrix& A,
               std::span<const Real> b,
               std::span<const Real> x) {
    std::vector<Real> r(b.size());
    mac::la::residual(A, b, x, r);
    Real rn = mac::la::nrm2(r);
    Real bn = mac::la::nrm2(b);
    return bn == 0.0 ? rn : rn / bn;
}

Real rel_err(std::span<const Real> x, std::span<const Real> x_true) {
    Real num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        Real d = x[i] - x_true[i];
        num += d * d;
        den += x_true[i] * x_true[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

} // anon

TEST_CASE("CSR symmetry + diag", "[la][csr]") {
    auto sys = make_spd(20, 4, /*alpha=*/2.0, /*seed=*/1);
    REQUIRE(sys.A.is_symmetric(1e-12));
    for (Index i = 0; i < sys.A.rows(); ++i) {
        REQUIRE(sys.A.diag(i) == Catch::Approx(sys.Adense(i, i)).margin(1e-12));
    }
}

TEST_CASE("SpMV matches Eigen", "[la][spmv]") {
    auto sys = make_spd(40, 5, 1.0, 7);
    std::vector<Real> x(40), y(40);
    std::mt19937_64 rng(42);
    std::normal_distribution<Real> nd(0, 1);
    for (auto& v : x) v = nd(rng);

    mac::la::spmv(sys.A, x, y);

    Eigen::VectorXd xv(40); for (Index i = 0; i < 40; ++i) xv[i] = x[i];
    Eigen::VectorXd yv = sys.Adense * xv;
    for (Index i = 0; i < 40; ++i) {
        REQUIRE(y[i] == Catch::Approx(yv[i]).margin(1e-12));
    }
}

TEST_CASE("Dense Cholesky solves SPD to 1e-12", "[direct][cholesky]") {
    auto sys = make_spd(30, 5, 1.0, 11);
    std::vector<Real> x(30, 0.0);
    mac::solvers::DenseCholesky chol(sys.A);
    chol.solve(sys.b, x);

    REQUIRE(rel_resid(sys.A, sys.b, x) <= 1e-12);
    REQUIRE(rel_err(x, sys.x_true) <= 1e-10);
}

namespace {
using SolverFn =
    mac::solvers::IterativeResult (*)(const CsrMatrix&,
                                      std::span<const Real>,
                                      std::span<Real>,
                                      const mac::solvers::IterativeOpts&);

void check_well_conditioned(const char* name, SolverFn fn) {
    auto sys = make_spd(40, 4, /*alpha=*/4.0, /*seed=*/3);  // well-cond
    std::vector<Real> x(40, 0.0);
    mac::solvers::IterativeOpts opts;
    opts.rtol     = 1e-8;
    opts.max_iter = 5000;
    auto res = fn(sys.A, sys.b, x, opts);
    INFO(name << ": iters=" << res.iters
              << " rresid=" << res.final_rresid
              << " conv=" << res.converged);
    REQUIRE(res.converged);
    REQUIRE(res.final_rresid <= 1e-8);
    REQUIRE(rel_err(x, sys.x_true) <= 1e-6);
}
} // anon

TEST_CASE("Jacobi well-cond",       "[iterative][well-cond][jacobi]") {
    check_well_conditioned("jacobi", &mac::solvers::jacobi);
}
TEST_CASE("Gauss-Seidel well-cond", "[iterative][well-cond][gs]") {
    check_well_conditioned("gauss_seidel", &mac::solvers::gauss_seidel);
}
TEST_CASE("RB Gauss-Seidel well-cond", "[iterative][well-cond][rbgs]") {
    check_well_conditioned("rb_gauss_seidel", &mac::solvers::rb_gauss_seidel);
}
TEST_CASE("CG well-cond",           "[iterative][well-cond][cg]") {
    check_well_conditioned("cg", &mac::solvers::cg);
}
TEST_CASE("PCG-Jacobi well-cond",   "[iterative][well-cond][pcg]") {
    check_well_conditioned("pcg_jacobi", &mac::solvers::pcg_jacobi);
}

TEST_CASE("CG matches Eigen on small SPD", "[iterative][cg][oracle]") {
    auto sys = make_spd(25, 4, 2.0, 19);
    std::vector<Real> x(25, 0.0);
    mac::solvers::IterativeOpts opts;
    opts.rtol = 1e-10;
    opts.max_iter = 2000;
    auto res = mac::solvers::cg(sys.A, sys.b, x, opts);
    REQUIRE(res.converged);

    Eigen::VectorXd bv(25); for (Index i = 0; i < 25; ++i) bv[i] = sys.b[i];
    Eigen::VectorXd x_oracle = sys.Adense.llt().solve(bv);
    for (Index i = 0; i < 25; ++i) {
        REQUIRE(x[i] == Catch::Approx(x_oracle[i]).margin(1e-6));
    }
}

// Conditioning sweep: behavior, no pass/fail.
TEST_CASE("CG conditioning behavior log", "[iterative][cg][conditioning][!mayfail]") {
    for (Real alpha : {1.0, 1e-2, 1e-4, 1e-6}) {
        auto sys = make_spd(60, 4, alpha, 31);
        std::vector<Real> x(60, 0.0);
        mac::solvers::IterativeOpts opts;
        opts.rtol = 1e-8;
        opts.max_iter = 2000;
        auto res = mac::solvers::cg(sys.A, sys.b, x, opts);
        INFO("alpha=" << alpha << " iters=" << res.iters
             << " rresid=" << res.final_rresid
             << " conv=" << res.converged);
        // No REQUIRE — record only.
        SUCCEED("recorded");
    }
}
