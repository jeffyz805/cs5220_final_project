// Synthetic SPD solver benchmark driver (serial + MPI).
//
// Serial solvers (no MPI launcher needed):
//   solver_bench --N 10000 --solver cg
//   --solver in {jacobi, gs, rbgs, cg, pcg, dense_chol}
//
// Distributed solvers (run under srun/mpirun):
//   srun -n 64 solver_bench --N 1000000 --solver dist_cg --csv out.csv
//   --solver in {dist_cg, dist_pcg, dist_jacobi, dist_block_gs}
//
// Distributed mode auto-detected by solver name prefix `dist_`. Rank 0 alone
// emits the CSV row to keep output uncluttered.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "bench/spd_generator.h"
#include "la/blas.h"
#include "solvers/direct/dense_cholesky.h"
#include "solvers/iterative/iterative.h"

#if MAC_USE_MPI
#include <mpi.h>
#include "solvers/mpi/dist_solvers.h"
#include "solvers/mpi/distributed_blas.h"
#include "solvers/mpi/distributed_csr.h"
#include "solvers/mpi/partition.h"
#endif

namespace {

struct Args {
    int           N            = 1000;
    int           nnz_per_row  = 8;
    double        alpha        = 1.0;
    double        rtol         = 1e-8;
    int           max_iter     = 5000;
    std::uint64_t seed         = 0xC5220ULL;
    std::string   solver       = "cg";
    std::string   csv          = "";
    std::string   tag          = "";
};

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --N <n> --solver <name> [opts]\n"
        "  serial: jacobi gs rbgs cg pcg dense_chol\n"
        "  dist:   dist_cg dist_pcg dist_jacobi dist_block_gs\n", p);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", opt);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (k == "--N")            a.N = std::atoi(next(k.c_str()));
        else if (k == "--nnz_per_row")  a.nnz_per_row = std::atoi(next(k.c_str()));
        else if (k == "--alpha")        a.alpha = std::atof(next(k.c_str()));
        else if (k == "--rtol")         a.rtol  = std::atof(next(k.c_str()));
        else if (k == "--max_iter")     a.max_iter = std::atoi(next(k.c_str()));
        else if (k == "--seed")         a.seed = std::strtoull(next(k.c_str()), nullptr, 0);
        else if (k == "--solver")       a.solver = next(k.c_str());
        else if (k == "--csv")          a.csv = next(k.c_str());
        else if (k == "--tag")          a.tag = next(k.c_str());
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return false; }
    }
    return true;
}

double rel_err(std::span<const double> x, std::span<const double> x_true) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        double d = x[i] - x_true[i];
        num += d * d;
        den += x_true[i] * x_true[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

bool starts_with(const std::string& s, std::string_view p) {
    return s.size() >= p.size() && std::string_view(s).substr(0, p.size()) == p;
}

struct CommBreakdown {
    double t_alltoallv  = 0.0;
    double t_allreduce  = 0.0;
    double t_local_spmv = 0.0;
    double t_pack       = 0.0;
    std::size_t bytes_alltoallv = 0;
    std::size_t bytes_allreduce = 0;
    std::size_t n_alltoallv     = 0;
    std::size_t n_allreduce     = 0;
};

void emit_csv(const Args& a, const std::string& mode,
              int ranks, int iters, int converged, int stagnated, int diverged,
              double rresid, double err, double t_gen, double t_solve,
              const CommBreakdown& cb) {
    auto write = [&](std::FILE* f, bool header) {
        if (header) {
            std::fprintf(f, "tag,solver,mode,ranks,N,nnz_per_row,alpha,rtol,max_iter,"
                            "iters,converged,stagnated,diverged,"
                            "rresid,err,t_gen,t_solve,"
                            "t_alltoallv,t_allreduce,t_local_spmv,t_pack,"
                            "bytes_alltoallv,bytes_allreduce,n_alltoallv,n_allreduce\n");
        }
        std::fprintf(f,
            "%s,%s,%s,%d,%d,%d,%.6g,%.6g,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,"
            "%.6e,%.6e,%.6e,%.6e,%zu,%zu,%zu,%zu\n",
            a.tag.c_str(), a.solver.c_str(), mode.c_str(), ranks,
            a.N, a.nnz_per_row, a.alpha, a.rtol, a.max_iter,
            iters, converged, stagnated, diverged,
            rresid, err, t_gen, t_solve,
            cb.t_alltoallv, cb.t_allreduce, cb.t_local_spmv, cb.t_pack,
            cb.bytes_alltoallv, cb.bytes_allreduce, cb.n_alltoallv, cb.n_allreduce);
    };
    if (a.csv.empty()) {
        write(stdout, true);
    } else {
        bool exists = std::ifstream(a.csv).good();
        std::FILE* f = std::fopen(a.csv.c_str(), "a");
        if (!f) { std::perror(a.csv.c_str()); std::exit(1); }
        write(f, !exists);
        std::fclose(f);
    }
}

int run_serial(const Args& a) {
    namespace clk = std::chrono;
    using mac::la::Real;

    mac::bench::SpdGenOpts gopts;
    gopts.n           = a.N;
    gopts.nnz_per_row = a.nnz_per_row;
    gopts.alpha       = a.alpha;
    gopts.seed        = a.seed;

    auto t_gen0 = clk::steady_clock::now();
    auto A = mac::bench::generate_spd(gopts);
    auto rhs = mac::bench::make_rhs(A, a.seed ^ 0x1u);
    double t_gen = clk::duration<double>(clk::steady_clock::now() - t_gen0).count();

    std::vector<Real> x(a.N, 0.0);
    mac::solvers::IterativeOpts iopts;
    iopts.rtol = a.rtol;
    iopts.max_iter = a.max_iter;

    mac::solvers::IterativeResult ires{};
    auto t0 = clk::steady_clock::now();
    if      (a.solver == "jacobi") ires = mac::solvers::jacobi      (A, rhs.b, x, iopts);
    else if (a.solver == "gs")     ires = mac::solvers::gauss_seidel(A, rhs.b, x, iopts);
    else if (a.solver == "rbgs")   ires = mac::solvers::rb_gauss_seidel(A, rhs.b, x, iopts);
    else if (a.solver == "cg")     ires = mac::solvers::cg          (A, rhs.b, x, iopts);
    else if (a.solver == "pcg")    ires = mac::solvers::pcg_jacobi  (A, rhs.b, x, iopts);
    else if (a.solver == "dense_chol") {
        mac::solvers::DenseCholesky chol(A);
        chol.solve(rhs.b, x);
    } else {
        std::fprintf(stderr, "unknown serial solver: %s\n", a.solver.c_str());
        return 2;
    }
    double t_solve = clk::duration<double>(clk::steady_clock::now() - t0).count();

    std::vector<Real> r(a.N);
    mac::la::residual(A, rhs.b, x, r);
    double rresid = mac::la::nrm2(r) / std::max(mac::la::nrm2(rhs.b), 1e-300);
    double err = rel_err(x, rhs.x_true);

    emit_csv(a, "serial", 1, ires.iters,
             (int)ires.converged, (int)ires.stagnated, (int)ires.diverged,
             rresid, err, t_gen, t_solve, CommBreakdown{});
    return 0;
}

#if MAC_USE_MPI

int run_mpi(const Args& a) {
    namespace clk = std::chrono;
    using mac::la::Real;
    using namespace mac::solvers::mpi;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // For now: rank 0 generates the global system and scatters. Scalable
    // to N up to ~1e6 on a Perlmutter compute node.
    mac::la::CsrMatrix A_global;
    std::vector<Real> b_global;
    if (rank == 0) {
        mac::bench::SpdGenOpts gopts;
        gopts.n           = a.N;
        gopts.nnz_per_row = a.nnz_per_row;
        gopts.alpha       = a.alpha;
        gopts.seed        = a.seed;
        A_global = mac::bench::generate_spd(gopts);
        auto rhs = mac::bench::make_rhs(A_global, a.seed ^ 0x1u);
        b_global = std::move(rhs.b);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_gen0 = clk::steady_clock::now();

    auto part = RowPartition::make_block(a.N, size);
    auto local = scatter_global_csr(MPI_COMM_WORLD, part, A_global);
    DistributedCsr A_dist(MPI_COMM_WORLD, part, std::move(local));

    // Scatter b.
    std::vector<int> counts(size), displs(size + 1, 0);
    for (int p = 0; p < size; ++p) counts[p] = static_cast<int>(part.local_rows(p));
    for (int p = 0; p < size; ++p) displs[p + 1] = displs[p] + counts[p];
    std::vector<Real> b_local(A_dist.n_local());
    MPI_Scatterv(rank == 0 ? b_global.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 b_local.data(), static_cast<int>(b_local.size()), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_gen = clk::duration<double>(clk::steady_clock::now() - t_gen0).count();

    std::vector<Real> x_local(A_dist.n_local(), 0.0);
    mac::solvers::IterativeOpts iopts;
    iopts.rtol     = a.rtol;
    iopts.max_iter = a.max_iter;

    // Reset comm-stat counters before the solve.
    A_dist.reset_stats();
    g_allreduce_stats.reset();

    mac::solvers::IterativeResult ires;
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = clk::steady_clock::now();

    if      (a.solver == "dist_cg")       ires = dist_cg          (A_dist, b_local, x_local, iopts);
    else if (a.solver == "dist_pcg")      ires = dist_pcg_jacobi  (A_dist, b_local, x_local, iopts);
    else if (a.solver == "dist_jacobi")   ires = dist_jacobi      (A_dist, b_local, x_local, iopts);
    else if (a.solver == "dist_block_gs") ires = dist_block_gs    (A_dist, b_local, x_local, iopts);
    else {
        if (rank == 0) std::fprintf(stderr, "unknown distributed solver: %s\n", a.solver.c_str());
        return 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_solve = clk::duration<double>(clk::steady_clock::now() - t0).count();

    // Final residual: ||b - A x|| / ||b|| globally.
    std::vector<Real> Ax_local(A_dist.n_local());
    A_dist.spmv(x_local, Ax_local);
    Real local_r2 = 0.0, local_b2 = 0.0;
    for (std::size_t i = 0; i < b_local.size(); ++i) {
        Real ri = b_local[i] - Ax_local[i];
        local_r2 += ri * ri;
        local_b2 += b_local[i] * b_local[i];
    }
    Real global_r2 = 0.0, global_b2 = 0.0;
    MPI_Allreduce(&local_r2, &global_r2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_b2, &global_b2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double rresid = std::sqrt(global_r2 / std::max(global_b2, 1e-300));

    // Comm-vs-compute breakdown. Reduce per-rank values to MAX (slowest rank
    // dominates wall time) and SUM (for byte counts).
    const auto& halo = A_dist.stats();
    double  local_t[4] = {halo.time_alltoallv, g_allreduce_stats.total_seconds,
                          halo.time_local_spmv, halo.time_pack};
    double  max_t[4]   = {0, 0, 0, 0};
    MPI_Reduce(local_t, max_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long local_b[4] = {
        (unsigned long long)halo.bytes_sent_total,
        (unsigned long long)g_allreduce_stats.bytes_sent,
        (unsigned long long)halo.spmv_calls,
        (unsigned long long)g_allreduce_stats.n_calls};
    unsigned long long sum_b[4] = {0, 0, 0, 0};
    MPI_Reduce(local_b, sum_b, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CommBreakdown cb;
        cb.t_alltoallv     = max_t[0];
        cb.t_allreduce     = max_t[1];
        cb.t_local_spmv    = max_t[2];
        cb.t_pack          = max_t[3];
        cb.bytes_alltoallv = static_cast<std::size_t>(sum_b[0]);
        cb.bytes_allreduce = static_cast<std::size_t>(sum_b[1]);
        cb.n_alltoallv     = static_cast<std::size_t>(sum_b[2]);
        cb.n_allreduce     = static_cast<std::size_t>(sum_b[3]);

        emit_csv(a, "mpi", size, ires.iters,
                 (int)ires.converged, (int)ires.stagnated, (int)ires.diverged,
                 rresid, /*err=*/-1.0, t_gen, t_solve, cb);
    }
    return 0;
}

#endif // MAC_USE_MPI

} // anon

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    bool dist = starts_with(a.solver, "dist_");

#if MAC_USE_MPI
    if (dist) {
        MPI_Init(&argc, &argv);
        int rc = run_mpi(a);
        MPI_Finalize();
        return rc;
    }
#else
    if (dist) {
        std::fprintf(stderr, "this build has USE_MPI=OFF; rebuild w/ -DUSE_MPI=ON\n");
        return 2;
    }
#endif
    return run_serial(a);
}
