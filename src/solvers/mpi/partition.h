#pragma once

#include <cassert>
#include <vector>

#include "la/sparse_matrix.h"

namespace mac::solvers::mpi {

// 1D row-block partition. Rank r owns rows [rank_starts[r], rank_starts[r+1]).
// Rows are split as evenly as possible (extras go to lower ranks).
struct RowPartition {
    la::Index n_global   = 0;
    int       n_ranks    = 1;
    std::vector<la::Index> rank_starts;  // size n_ranks + 1

    static RowPartition make_block(la::Index n_global, int n_ranks) {
        RowPartition p;
        p.n_global = n_global;
        p.n_ranks  = n_ranks;
        p.rank_starts.assign(n_ranks + 1, 0);
        la::Index base = n_global / n_ranks;
        la::Index rem  = n_global % n_ranks;
        for (int r = 0; r < n_ranks; ++r) {
            la::Index sz = base + (r < rem ? 1 : 0);
            p.rank_starts[r + 1] = p.rank_starts[r] + sz;
        }
        return p;
    }

    la::Index local_rows(int r) const {
        assert(0 <= r && r < n_ranks);
        return rank_starts[r + 1] - rank_starts[r];
    }
    la::Index row_start(int r) const { return rank_starts[r]; }

    // Owner of a global row index. Linear scan; partition is small.
    int owner(la::Index global_row) const {
        assert(global_row >= 0 && global_row < n_global);
        for (int r = 0; r < n_ranks; ++r) {
            if (global_row < rank_starts[r + 1]) return r;
        }
        return n_ranks - 1;  // unreachable
    }
};

} // namespace mac::solvers::mpi
