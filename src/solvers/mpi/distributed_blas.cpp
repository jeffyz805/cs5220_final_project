#include "solvers/mpi/distributed_blas.h"

#include <chrono>
#include <cmath>

#include "la/blas.h"

namespace mac::solvers::mpi {

thread_local AllreduceStats g_allreduce_stats{};

namespace {
inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
} // anon

la::Real dist_dot(MPI_Comm comm,
                  std::span<const la::Real> x_local,
                  std::span<const la::Real> y_local) {
    la::Real local = la::dot(x_local, y_local);
    double t0 = now_s();
    la::Real global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    double t1 = now_s();
    g_allreduce_stats.n_calls       += 1;
    g_allreduce_stats.total_seconds += (t1 - t0);
    g_allreduce_stats.bytes_sent    += sizeof(la::Real);
    return global;
}

la::Real dist_nrm2(MPI_Comm comm, std::span<const la::Real> x_local) {
    la::Real local = la::dot(x_local, x_local);
    double t0 = now_s();
    la::Real global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    double t1 = now_s();
    g_allreduce_stats.n_calls       += 1;
    g_allreduce_stats.total_seconds += (t1 - t0);
    g_allreduce_stats.bytes_sent    += sizeof(la::Real);
    return std::sqrt(global);
}

} // namespace mac::solvers::mpi
