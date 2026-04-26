#include "la/sparse_matrix.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mac::la {

Real CsrMatrix::diag(Index r) const {
    auto cols = row_cols(r);
    auto vals = row_vals(r);
    auto it = std::lower_bound(cols.begin(), cols.end(), r);
    if (it != cols.end() && *it == r) {
        return vals[std::distance(cols.begin(), it)];
    }
    return 0.0;
}

namespace {
// Sort + dedupe (col, val) pairs in-place; sums duplicates.
void canonicalize(std::vector<Index>& cols, std::vector<Real>& vals) {
    const std::size_t n = cols.size();
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    std::sort(perm.begin(), perm.end(),
              [&](std::size_t a, std::size_t b) { return cols[a] < cols[b]; });

    std::vector<Index> sc; sc.reserve(n);
    std::vector<Real>  sv; sv.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Index c = cols[perm[i]];
        Real  v = vals[perm[i]];
        if (!sc.empty() && sc.back() == c) sv.back() += v;
        else { sc.push_back(c); sv.push_back(v); }
    }
    cols.swap(sc);
    vals.swap(sv);
}
} // anon

CsrMatrix CsrMatrix::from_triplets(Index n_rows, Index n_cols,
                                   std::span<const Index> rows,
                                   std::span<const Index> cols,
                                   std::span<const Real>  vals) {
    assert(rows.size() == cols.size() && cols.size() == vals.size());

    CsrBuilder b(n_rows, n_cols);
    for (std::size_t k = 0; k < rows.size(); ++k) {
        b.push(rows[k], cols[k], vals[k]);
    }
    return b.finalize();
}

bool CsrMatrix::is_symmetric(Real tol) const {
    if (n_rows_ != n_cols_) return false;
    for (Index r = 0; r < n_rows_; ++r) {
        auto cs = row_cols(r);
        auto vs = row_vals(r);
        for (std::size_t k = 0; k < cs.size(); ++k) {
            Index c = cs[k];
            Real  v = vs[k];
            if (c == r) continue;
            // Look up A(c, r).
            auto cs2 = row_cols(c);
            auto vs2 = row_vals(c);
            auto it = std::lower_bound(cs2.begin(), cs2.end(), r);
            if (it == cs2.end() || *it != r) return false;
            Real v2 = vs2[std::distance(cs2.begin(), it)];
            if (std::abs(v - v2) > tol) return false;
        }
    }
    return true;
}

void CsrBuilder::set_row(Index r,
                         std::span<const Index> cols,
                         std::span<const Real>  vals) {
    assert(cols.size() == vals.size());
    auto& row = rows_[r];
    row.cols.assign(cols.begin(), cols.end());
    row.vals.assign(vals.begin(), vals.end());
}

CsrMatrix CsrBuilder::finalize() const {
    // Dedupe each row into a local buffer first, then assemble compact CSR.
    std::vector<std::vector<Index>> dc(n_rows_);
    std::vector<std::vector<Real>>  dv(n_rows_);
    for (Index r = 0; r < n_rows_; ++r) {
        dc[r] = rows_[r].cols;
        dv[r] = rows_[r].vals;
        canonicalize(dc[r], dv[r]);
    }

    CsrMatrix A(n_rows_, n_cols_);
    A.row_ptr_.assign(n_rows_ + 1, 0);
    for (Index r = 0; r < n_rows_; ++r) {
        A.row_ptr_[r + 1] = A.row_ptr_[r] + static_cast<Index>(dc[r].size());
    }
    A.col_ind_.resize(A.row_ptr_.back());
    A.values_.resize(A.row_ptr_.back());
    for (Index r = 0; r < n_rows_; ++r) {
        Index off = A.row_ptr_[r];
        std::copy(dc[r].begin(), dc[r].end(), A.col_ind_.begin() + off);
        std::copy(dv[r].begin(), dv[r].end(), A.values_.begin()  + off);
    }
    return A;
}

} // namespace mac::la
