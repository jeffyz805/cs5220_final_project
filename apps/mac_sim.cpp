// mac_sim — stylized coarse-grained complement-cascade Brownian-dynamics
// simulation. Drives the constraint solver under dynamic topology change as
// proteins bind into the membrane attack complex.
//
// CLI:
//   mac_sim --steps N [--seed S] [--solver pcg|cg|gs|dense_chol]
//           [--rtol 1e-8] [--max_iter 500]
//           [--n_C5b 5] [--n_C6 5] [--n_C7 5] [--n_C8 5] [--n_C9 5]
//           [--box 8.0] [--kT 1.0] [--dt 5e-3]
//           [--baumgarte 5.0] [--eps_reg 1e-6] [--k_conf 0.5]
//           [--r_bind 0.3]
//           [--dump_every K] [--out_dir results/mac_run]
//
// Outputs (in --out_dir):
//   metrics.csv : one row per step
//   traj.xyz    : multi-frame XYZ (one frame per --dump_every steps)
//   summary.txt : final state summary

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "mac/binding.h"
#include "mac/mac_world.h"
#include "mac/protein.h"
#include "mac/scenario.h"
#include "physics/integrator.h"

namespace {

struct Args {
    int    steps      = 1000;
    int    dump_every = 50;
    std::uint64_t seed = 0xCA5EEDULL;
    std::string solver = "pcg";
    double rtol        = 1e-8;
    int    max_iter    = 500;
    int    n_C5b = 5, n_C6 = 5, n_C7 = 5, n_C8 = 5, n_C9 = 5;
    double box  = 8.0;
    double kT   = 1.0;
    double dt   = 5e-3;
    double baumgarte = 5.0;
    double eps_reg   = 1e-6;
    double k_conf    = 0.5;
    double r_bind    = 0.3;
    std::string out_dir = "results/mac_run";
};

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --steps N [opts]\n"
        "  see source for full option list\n", p);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--steps")      a.steps = std::atoi(next());
        else if (k == "--dump_every") a.dump_every = std::atoi(next());
        else if (k == "--seed")       a.seed = std::strtoull(next(), nullptr, 0);
        else if (k == "--solver")     a.solver = next();
        else if (k == "--rtol")       a.rtol = std::atof(next());
        else if (k == "--max_iter")   a.max_iter = std::atoi(next());
        else if (k == "--n_C5b")      a.n_C5b = std::atoi(next());
        else if (k == "--n_C6")       a.n_C6 = std::atoi(next());
        else if (k == "--n_C7")       a.n_C7 = std::atoi(next());
        else if (k == "--n_C8")       a.n_C8 = std::atoi(next());
        else if (k == "--n_C9")       a.n_C9 = std::atoi(next());
        else if (k == "--box")        a.box = std::atof(next());
        else if (k == "--kT")         a.kT = std::atof(next());
        else if (k == "--dt")         a.dt = std::atof(next());
        else if (k == "--baumgarte")  a.baumgarte = std::atof(next());
        else if (k == "--eps_reg")    a.eps_reg = std::atof(next());
        else if (k == "--k_conf")     a.k_conf = std::atof(next());
        else if (k == "--r_bind")     a.r_bind = std::atof(next());
        else if (k == "--out_dir")    a.out_dir = next();
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return false; }
    }
    return true;
}

mac::physics::IntegratorOpts::Solver parse_solver(const std::string& s) {
    using S = mac::physics::IntegratorOpts::Solver;
    if (s == "cg")          return S::CG;
    if (s == "pcg")         return S::PCG;
    if (s == "gs")          return S::GS;
    if (s == "dense_chol")  return S::DenseChol;
    std::fprintf(stderr, "unknown solver: %s (valid: cg, pcg, gs, dense_chol)\n", s.c_str());
    std::exit(2);
}

void write_xyz_frame(std::FILE* f, const mac::macsim::MacWorld& mw, int step) {
    std::fprintf(f, "%d\n", mw.world.n_bodies());
    std::fprintf(f, "step=%d nbind=%d\n", step, mw.n_binding_events_total);
    for (int i = 0; i < mw.world.n_bodies(); ++i) {
        const auto& B = mw.world.bodies[i];
        auto name = mac::macsim::kind_name(mw.kind_of(i));
        std::fprintf(f, "%s %.6f %.6f %.6f\n",
                     std::string(name).c_str(), B.x.x, B.x.y, B.x.z);
    }
}

} // anon

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    std::filesystem::create_directories(a.out_dir);

    // Build scenario.
    mac::macsim::ScenarioOpts sopts;
    sopts.n_C5b = a.n_C5b; sopts.n_C6 = a.n_C6; sopts.n_C7 = a.n_C7;
    sopts.n_C8  = a.n_C8;  sopts.n_C9 = a.n_C9;
    sopts.box_size = a.box;
    sopts.kT = a.kT; sopts.dt = a.dt;
    sopts.eps_reg = a.eps_reg; sopts.baumgarte = a.baumgarte;
    sopts.k_conf = a.k_conf;
    sopts.seed = a.seed;
    auto mw = mac::macsim::make_scenario(sopts);

    mac::macsim::BindingOpts bopts;
    bopts.r_bind = a.r_bind;

    mac::physics::IntegratorOpts iopts;
    iopts.solver = parse_solver(a.solver);
    iopts.iter_opts.rtol     = a.rtol;
    iopts.iter_opts.max_iter = a.max_iter;
    iopts.seed = a.seed ^ 0xC0FFEEULL;

    std::mt19937_64 rng(iopts.seed);

    // Open outputs.
    std::FILE* csv = std::fopen((a.out_dir + "/metrics.csv").c_str(), "w");
    if (!csv) { std::perror("metrics.csv"); return 1; }
    std::fprintf(csv,
        "step,n_bodies,n_constraints,n_rows,bind_this_step,bind_total,"
        "iters,converged,rresid,t_assemble,t_solve,t_integrate\n");

    std::FILE* xyz = std::fopen((a.out_dir + "/traj.xyz").c_str(), "w");
    if (!xyz) { std::perror("traj.xyz"); return 1; }
    write_xyz_frame(xyz, mw, 0);

    std::printf("mac_sim: %d bodies, %d steps, solver=%s, dt=%.4g\n",
                mw.world.n_bodies(), a.steps, a.solver.c_str(), a.dt);

    auto t_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= a.steps; ++step) {
        // 1. Try to bind any eligible site pairs.
        mac::macsim::detect_and_bind(mw, bopts);

        // 2. Build external-force closure: confining well only.
        auto fext = [&](const mac::physics::World&, std::span<mac::la::Real> F){
            mac::macsim::apply_confining_force(mw, a.k_conf, F);
        };

        // 3. Integrate one Brownian step under active constraints.
        auto stats = mac::physics::step(mw.world, mw.constraints, fext, rng, iopts);

        // 4. Log + dump.
        std::fprintf(csv,
            "%d,%d,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n",
            step, mw.world.n_bodies(),
            (int)mw.constraints.size(), stats.n_rows,
            mw.n_bindings_this_step, mw.n_binding_events_total,
            stats.iters, (int)stats.converged, stats.final_rresid,
            stats.t_assemble, stats.t_solve, stats.t_integrate);

        if (a.dump_every > 0 && (step % a.dump_every == 0)) {
            write_xyz_frame(xyz, mw, step);
        }

        if (step % std::max(1, a.steps / 20) == 0) {
            std::printf("  step=%d cons=%zu bind=%d iters=%d rresid=%.2e\n",
                        step, mw.constraints.size(), mw.n_binding_events_total,
                        stats.iters, stats.final_rresid);
            std::fflush(stdout);
        }
    }

    double t_total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    std::fclose(csv);
    std::fclose(xyz);

    std::FILE* sum = std::fopen((a.out_dir + "/summary.txt").c_str(), "w");
    std::fprintf(sum, "mac_sim summary\n");
    std::fprintf(sum, "  bodies        : %d\n", mw.world.n_bodies());
    std::fprintf(sum, "  steps         : %d\n", a.steps);
    std::fprintf(sum, "  binding_events: %d\n", mw.n_binding_events_total);
    std::fprintf(sum, "  constraints   : %zu\n", mw.constraints.size());
    std::fprintf(sum, "  solver        : %s\n", a.solver.c_str());
    std::fprintf(sum, "  wall_time_s   : %.4f\n", t_total);
    std::fclose(sum);

    std::printf("done in %.2fs (%d binding events, %zu constraints)\n",
                t_total, mw.n_binding_events_total, mw.constraints.size());
    return 0;
}
