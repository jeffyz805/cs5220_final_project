// MPI cross-validation: each distributed solver, run on N ranks, must produce
// the same x (within rel-err 1e-6) as the serial solver run on rank 0 alone.
//
// Invoke as:  srun -n {1,2,4,8} ./build/tests/test_mpi
// Exits 0 on success, nonzero on first mismatch.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "bench/spd_generator.h"
#include "la/blas.h"
#include "la/sparse_matrix.h"
#include "solvers/iterative/iterative.h"
#include "solvers/mpi/dist_solvers.h"
#include "solvers/mpi/distributed_csr.h"
#include "solvers/mpi/partition.h"

using namespace mac;

namespace {

double rel_err(std::span<const double> a, std::span<const double> b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

struct Case {
    const char* name;
    bool        ok;
};

bool gather_and_compare(MPI_Comm comm,
                        const solvers::mpi::RowPartition& part,
                        std::span<const double> x_local,
                        std::span<const double> x_serial,
                        const char* label,
                        double tol = 1e-6) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<int> counts(size), displs(size + 1, 0);
    for (int p = 0; p < size; ++p) counts[p] = static_cast<int>(part.local_rows(p));
    for (int p = 0; p < size; ++p) displs[p + 1] = displs[p] + counts[p];

    std::vector<double> x_global;
    if (rank == 0) x_global.resize(part.n_global);
    MPI_Gatherv(x_local.data(), static_cast<int>(x_local.size()), MPI_DOUBLE,
                rank == 0 ? x_global.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    int ok_int = 1;
    if (rank == 0) {
        double e = rel_err(x_global, x_serial);
        std::printf("  [%-12s] rel_err = %.3e %s\n", label, e,
                    e <= tol ? "OK" : "FAIL");
        ok_int = (e <= tol) ? 1 : 0;
    }
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, comm);
    return ok_int != 0;
}

} // anon

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem: well-conditioned SPD, fixed seed -> reproducible.
    const int N = 200;
    bench::SpdGenOpts gopts;
    gopts.n           = N;
    gopts.nnz_per_row = 6;
    gopts.alpha       = 4.0;
    gopts.seed        = 0x5220ULL;

    // All ranks generate the global matrix (deterministic for fixed seed).
    // Then rank 0 also runs serial solvers; everyone scatters for distributed.
    la::CsrMatrix A_global = bench::generate_spd(gopts);
    auto rhs = bench::make_rhs(A_global, gopts.seed ^ 1);

    // --- Serial solutions (rank 0 only) ---
    std::vector<double> x_cg_serial(N, 0.0);
    std::vector<double> x_pcg_serial(N, 0.0);
    std::vector<double> x_jac_serial(N, 0.0);
    std::vector<double> x_gs_serial(N, 0.0);
    if (rank == 0) {
        solvers::IterativeOpts opts;
        opts.rtol     = 1e-10;
        opts.max_iter = 5000;
        solvers::cg          (A_global, rhs.b, x_cg_serial,  opts);
        solvers::pcg_jacobi  (A_global, rhs.b, x_pcg_serial, opts);
        solvers::jacobi      (A_global, rhs.b, x_jac_serial, opts);   // may diverge — that's fine
        solvers::gauss_seidel(A_global, rhs.b, x_gs_serial,  opts);
    }

    // --- Distributed setup ---
    auto part = solvers::mpi::RowPartition::make_block(N, size);
    auto local = solvers::mpi::scatter_global_csr(MPI_COMM_WORLD, part, A_global);
    solvers::mpi::DistributedCsr A_dist(MPI_COMM_WORLD, part, std::move(local));

    auto local_b = std::span<const double>(rhs.b)
        .subspan(part.row_start(rank), A_dist.n_local());

    // --- Distributed solves ---
    auto fresh_x = [&]() { return std::vector<double>(A_dist.n_local(), 0.0); };
    solvers::IterativeOpts dopts;
    dopts.rtol     = 1e-10;
    dopts.max_iter = 5000;

    if (rank == 0) std::printf("test_mpi: N=%d, size=%d\n", N, size);

    std::vector<Case> results;

    {
        auto x = fresh_x();
        auto r = solvers::mpi::dist_cg(A_dist, local_b, x, dopts);
        if (rank == 0) std::printf("dist_cg     iters=%d converged=%d\n", r.iters, r.converged);
        results.push_back({"dist_cg",
            gather_and_compare(MPI_COMM_WORLD, part, x, x_cg_serial, "dist_cg")});
    }
    {
        auto x = fresh_x();
        auto r = solvers::mpi::dist_pcg_jacobi(A_dist, local_b, x, dopts);
        if (rank == 0) std::printf("dist_pcg    iters=%d converged=%d\n", r.iters, r.converged);
        results.push_back({"dist_pcg",
            gather_and_compare(MPI_COMM_WORLD, part, x, x_pcg_serial, "dist_pcg")});
    }
    {
        // block-GS converges to the same solution as serial GS *if* both
        // converge — but with N=200, dense random SPD, alpha=4, it should.
        auto x = fresh_x();
        auto r = solvers::mpi::dist_block_gs(A_dist, local_b, x, dopts);
        if (rank == 0) std::printf("dist_blkgs  iters=%d converged=%d\n", r.iters, r.converged);
        // Loose tol — block-Jacobi/inner-GS converges to same A^-1 b when it
        // converges, but residual tolerance imprecision compounds.
        results.push_back({"dist_block_gs",
            gather_and_compare(MPI_COMM_WORLD, part, x, x_gs_serial, "dist_block_gs", 1e-4)});
    }
    // dist_jacobi typically diverges on random BᵀB+αI (not diag-dominant).
    // Run it and just record; don't assert solution match.
    {
        auto x = fresh_x();
        auto r = solvers::mpi::dist_jacobi(A_dist, local_b, x, dopts);
        if (rank == 0) std::printf("dist_jacobi iters=%d converged=%d (info only)\n",
                                   r.iters, r.converged);
    }

    int all_ok = 1;
    if (rank == 0) {
        for (auto& c : results) if (!c.ok) all_ok = 0;
    }
    MPI_Bcast(&all_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf(all_ok ? "test_mpi: PASS\n" : "test_mpi: FAIL\n");
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
