#pragma once

#include <span>
#include <vector>

#include "la/sparse_matrix.h"

namespace mac::solvers {

// Densified Cholesky baseline. Materializes A as a dense N×N row-major matrix,
// then performs a textbook in-place Cholesky factorization (lower-triangular L,
// L L^T = A) followed by forward + backward substitution.
//
// Intended only for small N (bench oracle and direct-method baseline). For
// large sparse systems use the CHOLMOD wrapper (USE_CHOLMOD=ON).

class DenseCholesky {
public:
    // Factorize A (must be SPD). Throws std::runtime_error on non-pos-def diag.
    explicit DenseCholesky(const la::CsrMatrix& A);

    // Solve A x = b in place (b in, x out). x.size() == n.
    void solve(std::span<const la::Real> b, std::span<la::Real> x) const;

    la::Index n() const { return n_; }

private:
    la::Index n_;
    std::vector<la::Real> L_;  // row-major n×n; only lower triangle valid
};

} // namespace mac::solvers
