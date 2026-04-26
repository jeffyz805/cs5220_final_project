#include "la/blas.h"

#include <cassert>
#include <cmath>

#if MAC_USE_AVX2
#include <immintrin.h>
#endif

namespace mac::la {

#if MAC_USE_AVX2

// AVX2 (256-bit) double-precision: 4 lanes per vector.
// Hot loops are kept narrow on purpose; we let the compiler unroll under -O3.

void axpy(Real a, std::span<const Real> x, std::span<Real> y) {
    assert(x.size() == y.size());
    const std::size_t n = x.size();
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x.data() + i);
        __m256d vy = _mm256_loadu_pd(y.data() + i);
        vy = _mm256_fmadd_pd(va, vx, vy);
        _mm256_storeu_pd(y.data() + i, vy);
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

Real dot(std::span<const Real> x, std::span<const Real> y) {
    assert(x.size() == y.size());
    const std::size_t n = x.size();
    __m256d acc = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x.data() + i);
        __m256d vy = _mm256_loadu_pd(y.data() + i);
        acc = _mm256_fmadd_pd(vx, vy, acc);
    }
    alignas(32) double buf[4];
    _mm256_store_pd(buf, acc);
    Real s = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}

void scal(Real a, std::span<Real> x) {
    const std::size_t n = x.size();
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x.data() + i);
        vx = _mm256_mul_pd(vx, va);
        _mm256_storeu_pd(x.data() + i, vx);
    }
    for (; i < n; ++i) x[i] *= a;
}

void xpay(Real a, std::span<const Real> x, std::span<Real> y) {
    assert(x.size() == y.size());
    const std::size_t n = x.size();
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x.data() + i);
        __m256d vy = _mm256_loadu_pd(y.data() + i);
        vy = _mm256_fmadd_pd(va, vy, vx); // x + a*y
        _mm256_storeu_pd(y.data() + i, vy);
    }
    for (; i < n; ++i) y[i] = x[i] + a * y[i];
}

#else // scalar fallback (also active on arm64 dev box)

void axpy(Real a, std::span<const Real> x, std::span<Real> y) {
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] += a * x[i];
}

Real dot(std::span<const Real> x, std::span<const Real> y) {
    assert(x.size() == y.size());
    Real s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
    return s;
}

void scal(Real a, std::span<Real> x) {
    for (auto& v : x) v *= a;
}

void xpay(Real a, std::span<const Real> x, std::span<Real> y) {
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] + a * y[i];
}

#endif

Real nrm2(std::span<const Real> x) {
    return std::sqrt(dot(x, x));
}

void copy(std::span<const Real> x, std::span<Real> y) {
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i];
}

void spmv(const CsrMatrix& A, std::span<const Real> x, std::span<Real> y) {
    assert(static_cast<Index>(x.size()) == A.cols());
    assert(static_cast<Index>(y.size()) == A.rows());
    auto rp = A.row_ptr();
    auto ci = A.col_ind();
    auto vs = A.values();

#if MAC_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Index r = 0; r < A.rows(); ++r) {
        Real s = 0.0;
        Index a = rp[r], b = rp[r + 1];
        for (Index k = a; k < b; ++k) s += vs[k] * x[ci[k]];
        y[r] = s;
    }
}

void residual(const CsrMatrix& A,
              std::span<const Real> b,
              std::span<const Real> x,
              std::span<Real> r) {
    spmv(A, x, r);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
}

} // namespace mac::la
