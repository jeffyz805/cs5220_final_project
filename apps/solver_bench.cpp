// Synthetic SPD solver benchmark driver.
//
// Runs a single solver on a generated SPD system, measures wall time + iter
// count + final residual, dumps a one-line CSV record. Designed to be invoked
// many times by SLURM scripts (one row per srun).
//
// CLI:
//   solver_bench --N <int> --solver <name> [options]
// solvers: jacobi, gs, rbgs, cg, pcg, dense_chol
// options:
//   --nnz_per_row <int>  (default 8)
//   --alpha       <real> (default 1.0)
//   --rtol        <real> (default 1e-8)
//   --max_iter    <int>  (default 5000)
//   --seed        <int>  (default 0xC5220)
//   --csv         <path> (default stdout, append; header written if new)
//   --tag         <str>  (free-form label written into CSV)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "bench/spd_generator.h"
#include "la/blas.h"
#include "solvers/direct/dense_cholesky.h"
#include "solvers/iterative/iterative.h"

namespace {

struct Args {
    int         N            = 1000;
    int         nnz_per_row  = 8;
    double      alpha        = 1.0;
    double      rtol         = 1e-8;
    int         max_iter     = 5000;
    std::uint64_t seed       = 0xC5220ULL;
    std::string solver       = "cg";
    std::string csv          = "";
    std::string tag          = "";
};

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --N <n> --solver <jacobi|gs|rbgs|cg|pcg|dense_chol> "
        "[--nnz_per_row k] [--alpha a] [--rtol t] [--max_iter m] "
        "[--seed s] [--csv path] [--tag str]\n", p);
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
        if      (k == "--N")            a.N = std::atoi(next("--N"));
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

} // anon

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    namespace clk = std::chrono;
    using mac::la::Real;

    // 1. Generate SPD system.
    mac::bench::SpdGenOpts gopts;
    gopts.n           = a.N;
    gopts.nnz_per_row = a.nnz_per_row;
    gopts.alpha       = a.alpha;
    gopts.seed        = a.seed;

    auto t_gen0 = clk::steady_clock::now();
    auto A = mac::bench::generate_spd(gopts);
    auto rhs = mac::bench::make_rhs(A, a.seed ^ 0x1u);
    auto t_gen1 = clk::steady_clock::now();
    double t_gen = clk::duration<double>(t_gen1 - t_gen0).count();

    std::vector<Real> x(a.N, 0.0);

    // 2. Solve.
    mac::solvers::IterativeOpts iopts;
    iopts.rtol = a.rtol;
    iopts.max_iter = a.max_iter;

    mac::solvers::IterativeResult ires{};
    bool is_iterative = true;

    auto t0 = clk::steady_clock::now();

    if      (a.solver == "jacobi") ires = mac::solvers::jacobi      (A, rhs.b, x, iopts);
    else if (a.solver == "gs")     ires = mac::solvers::gauss_seidel(A, rhs.b, x, iopts);
    else if (a.solver == "rbgs")   ires = mac::solvers::rb_gauss_seidel(A, rhs.b, x, iopts);
    else if (a.solver == "cg")     ires = mac::solvers::cg          (A, rhs.b, x, iopts);
    else if (a.solver == "pcg")    ires = mac::solvers::pcg_jacobi  (A, rhs.b, x, iopts);
    else if (a.solver == "dense_chol") {
        is_iterative = false;
        mac::solvers::DenseCholesky chol(A);
        chol.solve(rhs.b, x);
    }
    else {
        std::fprintf(stderr, "unknown solver: %s\n", a.solver.c_str());
        return 2;
    }

    auto t1 = clk::steady_clock::now();
    double t_solve = clk::duration<double>(t1 - t0).count();

    // 3. Final residual + error.
    std::vector<Real> r(a.N);
    mac::la::residual(A, rhs.b, x, r);
    double rn = mac::la::nrm2(r);
    double bn = mac::la::nrm2(rhs.b);
    double rresid = bn > 0 ? rn / bn : rn;
    double err = rel_err(x, rhs.x_true);

    // 4. CSV emit.
    auto write = [&](std::FILE* f, bool header) {
        if (header) {
            std::fprintf(f, "tag,solver,N,nnz_per_row,alpha,rtol,max_iter,"
                            "iters,converged,stagnated,diverged,"
                            "rresid,err,t_gen,t_solve\n");
        }
        std::fprintf(f,
            "%s,%s,%d,%d,%.6g,%.6g,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n",
            a.tag.c_str(), a.solver.c_str(), a.N, a.nnz_per_row,
            a.alpha, a.rtol, a.max_iter,
            ires.iters,
            (int)ires.converged, (int)ires.stagnated, (int)ires.diverged,
            rresid, err, t_gen, t_solve);
        (void)is_iterative;
    };

    if (a.csv.empty()) {
        write(stdout, true);
    } else {
        bool exists = std::ifstream(a.csv).good();
        std::FILE* f = std::fopen(a.csv.c_str(), "a");
        if (!f) { std::perror(a.csv.c_str()); return 1; }
        write(f, !exists);
        std::fclose(f);
    }
    return 0;
}
