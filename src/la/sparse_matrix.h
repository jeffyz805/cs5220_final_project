#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace mac::la {

using Index = std::int32_t;
using Real  = double;

// CSR sparse matrix.
//
// Two construction modes:
//   1. Bulk:  CsrMatrix::from_triplets(...)
//   2. Incremental builder:  CsrBuilder, supports add_row / set_row / clear_row
//      so the constraint assembler in P4/P5 can mutate topology cheaply.
//
// Storage layout:
//   row_ptr_  size n_rows + 1, monotonic
//   col_ind_  size nnz
//   values_   size nnz
//
// Conventions:
//   - Indices within a row are kept sorted ascending. Solvers and SpMV
//     assume sorted columns.
class CsrMatrix {
public:
    CsrMatrix() = default;
    CsrMatrix(Index n_rows, Index n_cols)
        : n_rows_(n_rows), n_cols_(n_cols), row_ptr_(n_rows + 1, 0) {}

    Index rows() const { return n_rows_; }
    Index cols() const { return n_cols_; }
    Index nnz()  const { return row_ptr_.empty() ? 0 : row_ptr_.back(); }

    std::span<const Index> row_ptr() const { return row_ptr_; }
    std::span<const Index> col_ind() const { return col_ind_; }
    std::span<const Real>  values()  const { return values_; }
    std::span<Real>        values_mut()    { return values_; }

    // Direct row access — read-only.
    std::span<const Index> row_cols(Index r) const {
        Index a = row_ptr_[r], b = row_ptr_[r + 1];
        return {col_ind_.data() + a, static_cast<size_t>(b - a)};
    }
    std::span<const Real> row_vals(Index r) const {
        Index a = row_ptr_[r], b = row_ptr_[r + 1];
        return {values_.data() + a, static_cast<size_t>(b - a)};
    }

    // Diagonal extraction (returns 0 if missing). Linear scan within the row.
    Real diag(Index r) const;

    // Mutable views for fast in-place updates of nonzero values
    // (column structure unchanged).
    std::span<Real> row_vals_mut(Index r) {
        Index a = row_ptr_[r], b = row_ptr_[r + 1];
        return {values_.data() + a, static_cast<size_t>(b - a)};
    }

    // Build from (i, j, v) triplets. Duplicates summed.
    static CsrMatrix from_triplets(Index n_rows, Index n_cols,
                                   std::span<const Index> rows,
                                   std::span<const Index> cols,
                                   std::span<const Real>  vals);

    // Symmetry check (structural + numerical) within tol.
    bool is_symmetric(Real tol = 1e-12) const;

private:
    friend class CsrBuilder;

    Index n_rows_ = 0;
    Index n_cols_ = 0;
    std::vector<Index> row_ptr_;
    std::vector<Index> col_ind_;
    std::vector<Real>  values_;
};

// Incremental row-oriented builder. Cheap row mutations; finalize() emits a
// CsrMatrix in canonical form (sorted, deduplicated columns per row).
class CsrBuilder {
public:
    CsrBuilder(Index n_rows, Index n_cols)
        : n_rows_(n_rows), n_cols_(n_cols), rows_(n_rows) {}

    void clear_row(Index r) { rows_[r].cols.clear(); rows_[r].vals.clear(); }

    // Append a single (col, val) entry to row r. Duplicates are summed at
    // finalize() time.
    void push(Index r, Index c, Real v) {
        assert(0 <= r && r < n_rows_);
        assert(0 <= c && c < n_cols_);
        rows_[r].cols.push_back(c);
        rows_[r].vals.push_back(v);
    }

    // Replace row r with the given (sorted or unsorted) entries.
    void set_row(Index r, std::span<const Index> cols,
                          std::span<const Real>  vals);

    Index rows() const { return n_rows_; }
    Index cols() const { return n_cols_; }

    CsrMatrix finalize() const;

private:
    struct Row { std::vector<Index> cols; std::vector<Real> vals; };
    Index n_rows_, n_cols_;
    std::vector<Row> rows_;
};

} // namespace mac::la
