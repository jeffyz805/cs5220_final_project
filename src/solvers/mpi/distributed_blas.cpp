#include "solvers/mpi/distributed_blas.h"

#include <cmath>

#include "la/blas.h"

namespace mac::solvers::mpi {

la::Real dist_dot(MPI_Comm comm,
                  std::span<const la::Real> x_local,
                  std::span<const la::Real> y_local) {
    la::Real local = la::dot(x_local, y_local);
    la::Real global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

la::Real dist_nrm2(MPI_Comm comm, std::span<const la::Real> x_local) {
    la::Real local = la::dot(x_local, x_local);
    la::Real global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global);
}

} // namespace mac::solvers::mpi
