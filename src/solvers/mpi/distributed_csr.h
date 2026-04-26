#pragma once

#include <mpi.h>

#include <cstdint>
#include <span>
#include <vector>

#include "la/sparse_matrix.h"
#include "solvers/mpi/partition.h"

namespace mac::solvers::mpi {

// Distributed CSR matrix on a 1D row-block partition.
//
// Construction takes a CsrMatrix whose rows are the local owned rows but
// whose column indices are in *global* space [0, n_global). On construction
// we:
//   1. Identify all unique external (non-owned) column indices.
//   2. Assign each a "halo slot" in [n_local, n_local + n_halo).
//   3. Renumber local CSR column indices: owned cols -> [0, n_local),
//      external cols -> halo slot.
//   4. Build per-neighbor send/recv lists (Alltoall + Alltoallv setup).
//
// SpMV reuses these lists every call: pack send buf -> Alltoallv -> recv buf
// -> local SpMV against [x_local; recv_buf].
class DistributedCsr {
public:
    // local_global_cols.rows() must == part.local_rows(rank).
    // local_global_cols.cols() must == part.n_global.
    DistributedCsr(MPI_Comm comm,
                   const RowPartition& part,
                   la::CsrMatrix local_global_cols);

    void spmv(std::span<const la::Real> x_local,
              std::span<la::Real>       y_local) const;

    // Halo exchange only: fetch current values of remote columns into the
    // internal halo buffer. After this, halo_buf() returns those values.
    // Useful for solvers (e.g. block-GS) that need the halo separately from
    // a SpMV. Also called automatically inside spmv().
    void fetch_halo(std::span<const la::Real> x_local) const;

    // Halo buffer, valid after fetch_halo() or spmv(). Length n_halo().
    std::span<const la::Real> halo_buf() const;

    // Owned diagonal entry at local row i (returns 0 if missing).
    la::Real diag_local(la::Index local_row) const;

    int       rank()       const { return rank_; }
    int       size()       const { return size_; }
    MPI_Comm  comm()       const { return comm_; }
    la::Index n_local()    const { return n_local_; }
    la::Index n_halo()     const { return n_halo_; }
    la::Index n_global()   const { return part_.n_global; }
    la::Index row_start()  const { return part_.row_start(rank_); }

    const RowPartition&    partition() const { return part_; }
    const la::CsrMatrix&   local()     const { return local_; }

    // Communication accounting (for breakdown plots).
    struct CommStats {
        std::size_t spmv_calls       = 0;
        std::size_t bytes_sent_total = 0;
        double      time_pack        = 0.0;
        double      time_alltoallv   = 0.0;
        double      time_local_spmv  = 0.0;
    };
    const CommStats& stats() const { return stats_; }
    void reset_stats() const { stats_ = CommStats{}; }

private:
    MPI_Comm     comm_;
    RowPartition part_;
    int          rank_ = 0;
    int          size_ = 1;

    la::Index    n_local_ = 0;
    la::Index    n_halo_  = 0;

    la::CsrMatrix local_;  // columns renumbered into extended-local space

    // Halo exchange tables. Both send/recv use Alltoallv, so we keep
    // per-rank counts/displs of size `size_`.
    std::vector<int>          send_counts_;   // [size_]
    std::vector<int>          send_displs_;   // [size_+1]
    std::vector<la::Index>    send_indices_;  // local row indices to ship out
    std::vector<int>          recv_counts_;   // [size_]
    std::vector<int>          recv_displs_;   // [size_+1]
    // recv_buf_ comes after x_local in the extended view.

    mutable std::vector<la::Real> send_buf_;
    mutable std::vector<la::Real> x_ext_;     // [x_local | recv_buf]
    mutable CommStats             stats_;
};

// Convenience: scatter a global CsrMatrix from rank 0 to all ranks. Each rank
// returns its slice with global column indices (ready for DistributedCsr).
//
// On non-root ranks, `global` may be empty; only rank 0's matrix is read.
la::CsrMatrix scatter_global_csr(MPI_Comm comm,
                                 const RowPartition& part,
                                 const la::CsrMatrix& global);

} // namespace mac::solvers::mpi
