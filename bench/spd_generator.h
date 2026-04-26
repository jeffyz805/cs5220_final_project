#pragma once

#include <cstdint>

#include "la/sparse_matrix.h"

namespace mac::bench {

// SPD matrix generator: A = B^T B + alpha*I where B is sparse Gaussian.
//
// nnz_per_row of B (not A) controls density; alpha controls conditioning
// (smaller alpha => larger condition number).
//
// For benchmark sizes (N up to 1e6), B^T B is materialized via a
// gather-by-column sweep over B without densifying. CSR is returned.
struct SpdGenOpts {
    la::Index     n            = 1000;
    int           nnz_per_row  = 8;
    la::Real      alpha        = 1.0;
    std::uint64_t seed         = 0xC5220ULL;
};

la::CsrMatrix generate_spd(const SpdGenOpts& opts);

// Convenience: pick a known x_true and produce b = A x_true.
struct Rhs { std::vector<la::Real> b, x_true; };
Rhs make_rhs(const la::CsrMatrix& A, std::uint64_t seed);

} // namespace mac::bench
