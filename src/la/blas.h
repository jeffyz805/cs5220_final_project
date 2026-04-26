#pragma once

#include <cstddef>
#include <span>

#include "la/sparse_matrix.h"

namespace mac::la {

// Level-1 vector kernels. AVX2-vectorized on x86_64 when MAC_USE_AVX2; scalar
// fallback otherwise (incl. arm64 dev box).
//
// All kernels take std::span and assume non-overlapping memory.

void   axpy(Real a, std::span<const Real> x, std::span<Real> y);   // y += a*x
Real   dot (std::span<const Real> x, std::span<const Real> y);
Real   nrm2(std::span<const Real> x);                              // ||x||_2
void   scal(Real a, std::span<Real> x);                            // x *= a
void   copy(std::span<const Real> x, std::span<Real> y);
void   xpay(Real a, std::span<const Real> x, std::span<Real> y);   // y = x + a*y

// y = A * x  (dense CSR SpMV, row-parallelizable; OpenMP-eligible).
// Caller must have y.size() == A.rows() and x.size() == A.cols().
void   spmv(const CsrMatrix& A, std::span<const Real> x, std::span<Real> y);

// r = b - A x
void   residual(const CsrMatrix& A,
                std::span<const Real> b,
                std::span<const Real> x,
                std::span<Real> r);

} // namespace mac::la
