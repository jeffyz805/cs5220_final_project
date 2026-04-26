#include "solvers/mpi/distributed_csr.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <unordered_map>

namespace mac::solvers::mpi {

using la::CsrMatrix;
using la::CsrBuilder;
using la::Index;
using la::Real;

namespace {
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
} // anon

DistributedCsr::DistributedCsr(MPI_Comm comm,
                               const RowPartition& part,
                               CsrMatrix local_global)
    : comm_(comm), part_(part) {
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
    assert(size_ == part_.n_ranks);
    n_local_ = part_.local_rows(rank_);
    assert(local_global.rows() == n_local_);
    assert(local_global.cols() == part_.n_global);

    const Index row_start = part_.row_start(rank_);
    const Index row_end   = row_start + n_local_;

    // Pass 1: enumerate unique external columns; collect by owning rank.
    // Keep them sorted (stable, ascending) so packing/unpacking matches.
    std::vector<std::vector<Index>> recv_by_rank(size_);
    {
        std::unordered_map<Index, char> seen;
        seen.reserve(static_cast<std::size_t>(local_global.nnz()));
        auto rp = local_global.row_ptr();
        auto ci = local_global.col_ind();
        for (Index r = 0; r < n_local_; ++r) {
            for (Index k = rp[r]; k < rp[r + 1]; ++k) {
                Index c = ci[k];
                if (c >= row_start && c < row_end) continue;  // owned
                if (seen.emplace(c, 1).second) {
                    int owner = part_.owner(c);
                    recv_by_rank[owner].push_back(c);
                }
            }
        }
        for (auto& v : recv_by_rank) std::sort(v.begin(), v.end());
    }

    // Halo slot assignment: external cols laid out by neighbor rank order, then
    // by ascending global col within each neighbor block.
    recv_counts_.assign(size_, 0);
    recv_displs_.assign(size_ + 1, 0);
    for (int p = 0; p < size_; ++p) {
        recv_counts_[p] = static_cast<int>(recv_by_rank[p].size());
        recv_displs_[p + 1] = recv_displs_[p] + recv_counts_[p];
    }
    n_halo_ = recv_displs_[size_];

    std::unordered_map<Index, Index> col_to_halo;  // global col -> halo slot
    col_to_halo.reserve(static_cast<std::size_t>(n_halo_));
    for (int p = 0; p < size_; ++p) {
        Index slot = recv_displs_[p];
        for (Index c : recv_by_rank[p]) col_to_halo.emplace(c, slot++);
    }

    // Pass 2: rewrite local CSR column indices into extended-local space.
    {
        CsrBuilder bld(n_local_, n_local_ + n_halo_);
        auto rp = local_global.row_ptr();
        auto ci = local_global.col_ind();
        auto vs = local_global.values();
        for (Index r = 0; r < n_local_; ++r) {
            for (Index k = rp[r]; k < rp[r + 1]; ++k) {
                Index c = ci[k];
                Index new_c;
                if (c >= row_start && c < row_end) new_c = c - row_start;
                else                                new_c = n_local_ + col_to_halo.at(c);
                bld.push(r, new_c, vs[k]);
            }
        }
        local_ = bld.finalize();
    }

    // Tell each peer how many of its columns we want, so they can size their
    // sends. Symmetric: their recv from us = our send to them.
    send_counts_.assign(size_, 0);
    MPI_Alltoall(recv_counts_.data(), 1, MPI_INT,
                 send_counts_.data(), 1, MPI_INT, comm_);
    send_displs_.assign(size_ + 1, 0);
    for (int p = 0; p < size_; ++p) {
        send_displs_[p + 1] = send_displs_[p] + send_counts_[p];
    }
    Index n_send = send_displs_[size_];

    // Send peers the actual global column indices we want (so they know which
    // of their owned rows to ship every SpMV).
    std::vector<Index> recv_index_flat(n_halo_);
    for (int p = 0; p < size_; ++p) {
        std::copy(recv_by_rank[p].begin(), recv_by_rank[p].end(),
                  recv_index_flat.begin() + recv_displs_[p]);
    }
    std::vector<Index> send_index_flat(n_send);

    static_assert(sizeof(Index) == 4, "MPI Alltoallv assumes 32-bit Index");
    MPI_Alltoallv(recv_index_flat.data(), recv_counts_.data(),
                  recv_displs_.data(), MPI_INT,
                  send_index_flat.data(), send_counts_.data(),
                  send_displs_.data(), MPI_INT, comm_);

    // Convert received global col indices into local row indices.
    send_indices_.resize(n_send);
    for (Index i = 0; i < n_send; ++i) {
        Index g = send_index_flat[i];
        assert(g >= row_start && g < row_end);
        send_indices_[i] = g - row_start;
    }

    send_buf_.resize(n_send);
    x_ext_.resize(static_cast<std::size_t>(n_local_ + n_halo_));
}

void DistributedCsr::fetch_halo(std::span<const Real> x_local) const {
    assert(static_cast<Index>(x_local.size()) == n_local_);

    double t0 = now_s();
    for (std::size_t i = 0; i < send_indices_.size(); ++i) {
        send_buf_[i] = x_local[send_indices_[i]];
    }
    for (Index i = 0; i < n_local_; ++i) x_ext_[i] = x_local[i];
    double t1 = now_s();

    Real* recv_dst = x_ext_.data() + n_local_;
    MPI_Alltoallv(send_buf_.data(),  send_counts_.data(),
                  send_displs_.data(), MPI_DOUBLE,
                  recv_dst,          recv_counts_.data(),
                  recv_displs_.data(), MPI_DOUBLE, comm_);
    double t2 = now_s();

    stats_.bytes_sent_total += send_buf_.size() * sizeof(Real);
    stats_.time_pack        += (t1 - t0);
    stats_.time_alltoallv   += (t2 - t1);
}

std::span<const Real> DistributedCsr::halo_buf() const {
    return {x_ext_.data() + n_local_, static_cast<std::size_t>(n_halo_)};
}

void DistributedCsr::spmv(std::span<const Real> x_local,
                          std::span<Real>       y_local) const {
    assert(static_cast<Index>(x_local.size()) == n_local_);
    assert(static_cast<Index>(y_local.size()) == n_local_);

    fetch_halo(x_local);

    double t2 = now_s();
    auto rp = local_.row_ptr();
    auto ci = local_.col_ind();
    auto vs = local_.values();
#if MAC_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Index r = 0; r < n_local_; ++r) {
        Real s = 0.0;
        Index a = rp[r], b = rp[r + 1];
        for (Index k = a; k < b; ++k) s += vs[k] * x_ext_[ci[k]];
        y_local[r] = s;
    }
    double t3 = now_s();

    stats_.spmv_calls      += 1;
    stats_.time_local_spmv += (t3 - t2);
}

Real DistributedCsr::diag_local(Index local_row) const {
    Index global_diag_col = local_row;  // owned diag is at extended local col == local_row
    auto cs = local_.row_cols(local_row);
    auto vs = local_.row_vals(local_row);
    auto it = std::lower_bound(cs.begin(), cs.end(), global_diag_col);
    if (it != cs.end() && *it == global_diag_col) {
        return vs[std::distance(cs.begin(), it)];
    }
    return 0.0;
}

CsrMatrix scatter_global_csr(MPI_Comm comm,
                             const RowPartition& part,
                             const CsrMatrix& global) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    assert(size == part.n_ranks);

    Index n_local = part.local_rows(rank);

    // Each rank gets a packed buffer: row_lengths[n_local] + col_inds + values.
    // Rank 0 builds and sends; others Recv.
    if (rank == 0) {
        assert(global.rows() == part.n_global);
        assert(global.cols() == part.n_global);
        auto rp = global.row_ptr();
        auto ci = global.col_ind();
        auto vs = global.values();

        for (int p = 1; p < size; ++p) {
            Index ps  = part.row_start(p);
            Index pe  = ps + part.local_rows(p);
            Index nzp = rp[pe] - rp[ps];
            std::vector<Index> row_len(part.local_rows(p));
            for (Index r = ps; r < pe; ++r) row_len[r - ps] = rp[r + 1] - rp[r];
            MPI_Send(row_len.data(), static_cast<int>(row_len.size()), MPI_INT, p, 0, comm);
            MPI_Send(ci.data() + rp[ps], static_cast<int>(nzp), MPI_INT,    p, 1, comm);
            MPI_Send(vs.data() + rp[ps], static_cast<int>(nzp), MPI_DOUBLE, p, 2, comm);
        }
        // Rank 0's own slice.
        Index ps = part.row_start(0);
        Index pe = ps + n_local;
        CsrBuilder bld(n_local, part.n_global);
        for (Index r = ps; r < pe; ++r) {
            for (Index k = rp[r]; k < rp[r + 1]; ++k) {
                bld.push(r - ps, ci[k], vs[k]);
            }
        }
        return bld.finalize();
    } else {
        std::vector<Index> row_len(n_local);
        MPI_Recv(row_len.data(), static_cast<int>(n_local), MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        Index nzp = 0;
        for (Index v : row_len) nzp += v;
        std::vector<Index> col(nzp);
        std::vector<Real>  val(nzp);
        MPI_Recv(col.data(), static_cast<int>(nzp), MPI_INT,    0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(val.data(), static_cast<int>(nzp), MPI_DOUBLE, 0, 2, comm, MPI_STATUS_IGNORE);

        CsrBuilder bld(n_local, part.n_global);
        Index off = 0;
        for (Index r = 0; r < n_local; ++r) {
            for (Index k = 0; k < row_len[r]; ++k) {
                bld.push(r, col[off + k], val[off + k]);
            }
            off += row_len[r];
        }
        return bld.finalize();
    }
}

} // namespace mac::solvers::mpi
