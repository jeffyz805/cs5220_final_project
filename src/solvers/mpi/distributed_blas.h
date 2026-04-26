#pragma once

#include <mpi.h>

#include <cstddef>
#include <span>

#include "la/sparse_matrix.h"

namespace mac::solvers::mpi {

// Distributed level-1 BLAS. All take rank-local std::span and reduce across
// the given communicator. Pointwise ops (axpy/scal/copy/xpay) are not here —
// reuse mac::la::axpy etc. directly on the local span.

la::Real dist_dot(MPI_Comm comm,
                  std::span<const la::Real> x_local,
                  std::span<const la::Real> y_local);

la::Real dist_nrm2(MPI_Comm comm, std::span<const la::Real> x_local);

// Per-thread Allreduce accounting. Updated by dist_dot/dist_nrm2. Reset at
// solver entry, read at solver exit. Bytes count payload only.
struct AllreduceStats {
    std::size_t n_calls       = 0;
    double      total_seconds = 0.0;
    std::size_t bytes_sent    = 0;
    void reset() { n_calls = 0; total_seconds = 0.0; bytes_sent = 0; }
};
extern thread_local AllreduceStats g_allreduce_stats;

} // namespace mac::solvers::mpi
