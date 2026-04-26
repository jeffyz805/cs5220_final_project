#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

#include "constraints/constraint.h"
#include "physics/rigid_body.h"
#include "solvers/iterative/iterative.h"

namespace mac::physics {

// External-force callback: fills F_body (size 6*n_bodies) with deterministic
// forces (potentials, walls, etc.). Brownian noise is added separately.
using ForceFn = std::function<void(const World&, std::span<la::Real> /*F_body*/)>;

struct IntegratorOpts {
    // Solver to use for the constraint system.
    enum class Solver { CG, PCG, GS, DenseChol } solver = Solver::PCG;
    solvers::IterativeOpts iter_opts{};   // rtol, max_iter for iterative solvers

    // Random seed (per-process). For multi-rank simulations use distinct seeds.
    std::uint64_t seed = 0xDEADBEEFCAFEBABEULL;

    // If true, do constraint solves; if false, free Brownian (no constraints).
    bool with_constraints = true;
};

struct StepStats {
    int  n_constraints  = 0;
    int  n_rows         = 0;
    int  iters          = 0;
    bool converged      = false;
    la::Real final_rresid = 0.0;
    double t_assemble   = 0.0;
    double t_solve      = 0.0;
    double t_integrate  = 0.0;
};

// Run a single Brownian step with optional equality constraints.
// `rng` is mutated. Returns per-step diagnostics.
StepStats step(World& world,
               const std::vector<constraints::Constraint>& cons,
               const ForceFn& f_ext,
               std::mt19937_64& rng,
               const IntegratorOpts& opts = {});

} // namespace mac::physics
