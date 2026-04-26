#pragma once

#include <mpi.h>

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

} // namespace mac::solvers::mpi
