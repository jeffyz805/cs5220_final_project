#include "solvers/direct/dense_cholesky.h"

#include <cmath>
#include <stdexcept>

namespace mac::solvers {

using la::Real;
using la::Index;

DenseCholesky::DenseCholesky(const la::CsrMatrix& A) : n_(A.rows()) {
    if (A.rows() != A.cols()) throw std::runtime_error("DenseCholesky: A not square");
    L_.assign(static_cast<std::size_t>(n_) * n_, 0.0);

    // Densify A.
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vs = A.values();
    for (Index r = 0; r < n_; ++r) {
        for (Index k = rp[r]; k < rp[r + 1]; ++k) {
            L_[static_cast<std::size_t>(r) * n_ + ci[k]] = vs[k];
        }
    }

    // In-place Cholesky-Banachiewicz (row-by-row).
    auto idx = [&](Index i, Index j) { return static_cast<std::size_t>(i) * n_ + j; };
    for (Index i = 0; i < n_; ++i) {
        for (Index j = 0; j <= i; ++j) {
            Real s = L_[idx(i, j)];
            for (Index k = 0; k < j; ++k) s -= L_[idx(i, k)] * L_[idx(j, k)];
            if (i == j) {
                if (s <= 0.0) throw std::runtime_error("DenseCholesky: matrix not SPD");
                L_[idx(i, j)] = std::sqrt(s);
            } else {
                L_[idx(i, j)] = s / L_[idx(j, j)];
            }
        }
        // Zero out strict upper triangle for cleanliness.
        for (Index j = i + 1; j < n_; ++j) L_[idx(i, j)] = 0.0;
    }
}

void DenseCholesky::solve(std::span<const Real> b, std::span<Real> x) const {
    auto idx = [&](Index i, Index j) { return static_cast<std::size_t>(i) * n_ + j; };
    std::vector<Real> y(n_);
    // Forward: L y = b.
    for (Index i = 0; i < n_; ++i) {
        Real s = b[i];
        for (Index k = 0; k < i; ++k) s -= L_[idx(i, k)] * y[k];
        y[i] = s / L_[idx(i, i)];
    }
    // Backward: L^T x = y.
    for (Index i = n_ - 1; i >= 0; --i) {
        Real s = y[i];
        for (Index k = i + 1; k < n_; ++k) s -= L_[idx(k, i)] * x[k];
        x[i] = s / L_[idx(i, i)];
    }
}

} // namespace mac::solvers
