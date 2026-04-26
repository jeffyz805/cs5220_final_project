#include "physics/integrator.h"

#include <chrono>
#include <cmath>

#include "constraints/assembler.h"
#include "la/blas.h"
#include "solvers/direct/dense_cholesky.h"
#include "solvers/iterative/iterative.h"

namespace mac::physics {

using la::Real;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Sample 6*n_bodies-vector of zero-mean Gaussian forces with std-dev tuned so
// that the resulting position/orientation increments give the correct
// Brownian variance for an overdamped step.
//
// Without constraints, the overdamped update is
//     Δq_b = (Δt / γ_b) F_b
// where γ_b is the per-DoF drag (γ_t for linear, γ_r for angular).
//
// We want <Δq Δq^T> = 2 k_B T Δt / γ_b for each DoF (Einstein relation), which
// implies the noise *force* at this DoF has variance 2 k_B T γ_b / Δt:
//     <F_b F_b^T> = 2 k_B T γ_b / Δt
// so we sample F = sqrt(2 k_B T γ_b / Δt) ξ, ξ ~ N(0, I).
//
// (This keeps the same code path as deterministic forces: just add noise to F
// before the constraint solve.)
void sample_brownian_force(const World& world,
                           std::mt19937_64& rng,
                           std::span<Real> F_body) {
    std::normal_distribution<Real> nd(0.0, 1.0);
    for (int b = 0; b < world.n_bodies(); ++b) {
        const auto& B = world.bodies[b];
        Real sigma_t = std::sqrt(2.0 * world.kT * B.gamma_t / world.dt);
        Real sigma_r = std::sqrt(2.0 * world.kT * B.gamma_r / world.dt);
        Real* f = F_body.data() + 6 * b;
        for (int k = 0; k < 3; ++k) f[k]     += sigma_t * nd(rng);
        for (int k = 0; k < 3; ++k) f[3 + k] += sigma_r * nd(rng);
    }
}

// Integrate a body forward by one Brownian step given the *total* generalized
// force on it (external + Brownian + constraint correction).
void integrate_body(RigidBody& B, const Real* F, Real dt) {
    // dx = (Δt / γ_t) F_lin
    Vec3 lin{F[0], F[1], F[2]};
    Vec3 ang{F[3], F[4], F[5]};

    Vec3 dx = lin * (dt / B.gamma_t);
    Vec3 om = ang * (1.0 / B.gamma_r);  // angular velocity ω = M_r^-1 τ

    B.x += dx;
    B.q  = B.q.integrate(om, dt);
}

} // anon

StepStats step(World& world,
               const std::vector<constraints::Constraint>& cons,
               const ForceFn& f_ext,
               std::mt19937_64& rng,
               const IntegratorOpts& opts) {
    const int n = world.n_bodies();
    std::vector<Real> F(6 * n, 0.0);

    // 1. External + Brownian forces.
    if (f_ext) f_ext(world, F);
    sample_brownian_force(world, rng, F);

    StepStats st;
    st.n_constraints = static_cast<int>(cons.size());

    // 2. Constraint solve (if any constraints + enabled).
    if (opts.with_constraints && !cons.empty()) {
        double t0 = now_s();
        auto sys = constraints::assemble(world, cons, F);
        double t1 = now_s();

        std::vector<Real> lambda(sys.n_rows, 0.0);

        solvers::IterativeResult ires{};
        if (opts.solver == IntegratorOpts::Solver::DenseChol) {
            solvers::DenseCholesky chol(sys.A);
            chol.solve(sys.rhs, lambda);
        } else {
            switch (opts.solver) {
                case IntegratorOpts::Solver::CG:
                    ires = solvers::cg(sys.A, sys.rhs, lambda, opts.iter_opts); break;
                case IntegratorOpts::Solver::PCG:
                    ires = solvers::pcg_jacobi(sys.A, sys.rhs, lambda, opts.iter_opts); break;
                case IntegratorOpts::Solver::GS:
                    ires = solvers::gauss_seidel(sys.A, sys.rhs, lambda, opts.iter_opts); break;
                default: break;
            }
        }
        double t2 = now_s();

        constraints::apply_jt_lambda(world, cons, sys, lambda, F);

        st.n_rows        = sys.n_rows;
        st.iters         = ires.iters;
        st.converged     = ires.converged;
        st.final_rresid  = ires.final_rresid;
        st.t_assemble    = t1 - t0;
        st.t_solve       = t2 - t1;
    }

    // 3. Position / orientation update.
    double t3 = now_s();
    for (int b = 0; b < n; ++b) {
        integrate_body(world.bodies[b], F.data() + 6 * b, world.dt);
    }
    st.t_integrate = now_s() - t3;
    return st;
}

} // namespace mac::physics
