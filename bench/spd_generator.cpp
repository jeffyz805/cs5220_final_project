#include "bench/spd_generator.h"

#include <random>
#include <unordered_map>
#include <vector>

#include "la/blas.h"

namespace mac::bench {

using la::CsrMatrix;
using la::CsrBuilder;
using la::Index;
using la::Real;

la::CsrMatrix generate_spd(const SpdGenOpts& opts) {
    const Index n = opts.n;
    const int nnz = opts.nnz_per_row;
    std::mt19937_64 rng(opts.seed);
    std::uniform_int_distribution<Index> col_dist(0, n - 1);
    std::normal_distribution<Real> val_dist(0.0, 1.0);

    // Store B as CSR-like row lists: B_rows[i] = vector<(col, val)>.
    std::vector<std::vector<std::pair<Index, Real>>> B_rows(n);
    // Also build column-indexed view of B for the BᵀB sweep:
    //   B_col[j] = list of (row i, value B[i,j])
    std::vector<std::vector<std::pair<Index, Real>>> B_col(n);
    for (Index i = 0; i < n; ++i) {
        for (int k = 0; k < nnz; ++k) {
            Index j = col_dist(rng);
            Real  v = val_dist(rng);
            B_rows[i].emplace_back(j, v);
            B_col [j].emplace_back(i, v);
        }
    }

    // For each row r of A = BᵀB:
    //   A[r, c] = sum_i B[i, r] * B[i, c]
    // i ranges over rows where B[i,r] != 0, i.e. B_col[r] entries.
    // For each such i, add B[i, r] * B[i, c] for every c in B_rows[i].
    CsrBuilder bld(n, n);
    std::unordered_map<Index, Real> rowmap;
    for (Index r = 0; r < n; ++r) {
        rowmap.clear();
        for (auto [i, b_ir] : B_col[r]) {
            for (auto [c, b_ic] : B_rows[i]) {
                rowmap[c] += b_ir * b_ic;
            }
        }
        // Add diagonal regularization alpha.
        rowmap[r] += opts.alpha;
        for (auto [c, v] : rowmap) bld.push(r, c, v);
    }
    return bld.finalize();
}

Rhs make_rhs(const CsrMatrix& A, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    Rhs out;
    out.x_true.resize(A.cols());
    for (auto& v : out.x_true) v = nd(rng);
    out.b.assign(A.rows(), 0.0);
    la::spmv(A, out.x_true, out.b);
    return out;
}

} // namespace mac::bench
