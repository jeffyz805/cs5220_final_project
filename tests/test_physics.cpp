// Physics-engine unit tests.
//
//   1. Vec3 / Quat sanity.
//   2. Constraint relaxation: perturbed 2-body distance constraint, no noise.
//      Violation must decay (Baumgarte stabilization).
//   3. Brownian MSD: 1-body free diffusion. <(Δx)^2> ≈ 2 D t = 2 kT t / γ.
//   4. Assembler sanity: A is symmetric, positive (eps regularization).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "constraints/assembler.h"
#include "constraints/constraint.h"
#include "la/blas.h"
#include "physics/integrator.h"
#include "physics/math3d.h"
#include "physics/rigid_body.h"

using mac::la::Real;
using mac::physics::Vec3;
using mac::physics::Quat;
using mac::physics::Mat3;
using mac::physics::World;
using mac::physics::RigidBody;
using mac::constraints::Constraint;
using mac::constraints::ConstraintKind;

TEST_CASE("Vec3 basic ops", "[physics][math]") {
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    REQUIRE(a.dot(b) == Catch::Approx(32.0));
    Vec3 c = a.cross(b);
    REQUIRE(c.x == Catch::Approx(-3.0));
    REQUIRE(c.y == Catch::Approx(6.0));
    REQUIRE(c.z == Catch::Approx(-3.0));
}

TEST_CASE("Quat rotates correctly", "[physics][math]") {
    // 90° about Z axis: (1,0,0) -> (0,1,0)
    Real h = std::sqrt(0.5);
    Quat q{h, 0, 0, h};  // cos(45), 0, 0, sin(45)
    Vec3 v = q.rotate({1, 0, 0});
    REQUIRE(v.x == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(v.y == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(v.z == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("Quat integrate small omega keeps unit norm", "[physics][math]") {
    Quat q = Quat::identity();
    Vec3 om{0.1, 0.2, -0.3};
    for (int i = 0; i < 1000; ++i) q = q.integrate(om, 1e-3);
    Real n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    REQUIRE(n == Catch::Approx(1.0).margin(1e-9));
}

// ---- Distance constraint relaxation (no noise) ----
TEST_CASE("Distance constraint relaxes a perturbed pair", "[physics][constraint]") {
    World w;
    w.kT       = 0.0;        // no Brownian force for this test
    w.dt       = 0.01;
    w.eps_reg  = 1e-8;
    // Baumgarte gain (decay rate, 1/time). Stable for β·dt « 2; β=10, dt=0.01
    // -> 10% reduction per step. After 200 steps (t=2), residual ≈ exp(-20).
    w.baumgarte = 10.0;

    w.bodies.resize(2);
    w.bodies[0].x = {0, 0, 0};
    w.bodies[1].x = {2.0, 0, 0};   // perturbed: target dist will be 1.0
    for (auto& B : w.bodies) { B.gamma_t = 1.0; B.gamma_r = 1.0; }

    Constraint c;
    c.kind = ConstraintKind::Distance;
    c.body_a = 0; c.body_b = 1;
    c.target_dist = 1.0;
    std::vector<Constraint> cons{c};

    std::mt19937_64 rng(7);
    mac::physics::IntegratorOpts opts;
    opts.iter_opts.rtol = 1e-10;
    opts.iter_opts.max_iter = 200;
    opts.solver = mac::physics::IntegratorOpts::Solver::PCG;

    Real init_violation;
    {
        std::vector<Real> C(1);
        mac::constraints::compute_violation(c, w.bodies[0], &w.bodies[1], C);
        init_violation = std::abs(C[0]);
    }
    REQUIRE(init_violation == Catch::Approx(1.0));

    for (int it = 0; it < 200; ++it) {
        mac::physics::step(w, cons, /*f_ext=*/{}, rng, opts);
    }

    std::vector<Real> C(1);
    mac::constraints::compute_violation(cons[0], w.bodies[0], &w.bodies[1], C);
    REQUIRE(std::abs(C[0]) < 1e-3);
}

// ---- Free Brownian: MSD test ----
//
// For a single isolated body in 3D under overdamped Langevin with diffusion
// coefficient D = kT / γ_t, <(Δx)²> = 6 D t (sum over 3 dims).
TEST_CASE("Brownian MSD matches 6 D t", "[physics][brownian]") {
    World w;
    w.kT       = 1.0;
    w.dt       = 0.005;
    w.eps_reg  = 1e-8;
    w.baumgarte = 0.0;

    RigidBody b;
    b.x = {0, 0, 0};
    b.gamma_t = 2.0;
    b.gamma_r = 2.0;
    w.bodies = {b};

    std::mt19937_64 rng(1234);
    mac::physics::IntegratorOpts opts;
    opts.with_constraints = false;
    opts.solver = mac::physics::IntegratorOpts::Solver::CG;

    const int n_traj = 2000;
    const int n_steps = 400;
    Real D = w.kT / b.gamma_t;
    Real t_total = w.dt * n_steps;
    Real expected = 6.0 * D * t_total;

    Real msd = 0.0;
    for (int k = 0; k < n_traj; ++k) {
        w.bodies[0].x = {0, 0, 0};
        w.bodies[0].q = Quat::identity();
        for (int s = 0; s < n_steps; ++s) {
            mac::physics::step(w, {}, /*f_ext=*/{}, rng, opts);
        }
        Vec3 x = w.bodies[0].x;
        msd += x.dot(x);
    }
    msd /= n_traj;

    // ~3% statistical error on 2000 trajectories.
    INFO("msd=" << msd << " expected=" << expected);
    REQUIRE(msd == Catch::Approx(expected).epsilon(0.08));
}

// ---- Assembler sanity ----
TEST_CASE("Assembler produces symmetric A with positive diagonal",
          "[physics][assembler]") {
    World w;
    w.dt = 0.01;
    w.eps_reg = 1e-6;
    w.bodies.resize(3);
    for (auto& B : w.bodies) { B.gamma_t = 1.0; B.gamma_r = 1.0; }
    w.bodies[0].x = {0, 0, 0};
    w.bodies[1].x = {1, 0, 0};
    w.bodies[2].x = {0, 1, 0};

    std::vector<Constraint> cons;
    Constraint c01;
    c01.kind = ConstraintKind::Distance;
    c01.body_a = 0; c01.body_b = 1; c01.target_dist = 1.0;
    cons.push_back(c01);
    Constraint c12;
    c12.kind = ConstraintKind::BallJoint;
    c12.body_a = 1; c12.body_b = 2;
    cons.push_back(c12);

    std::vector<Real> F(6 * w.n_bodies(), 0.0);
    auto sys = mac::constraints::assemble(w, cons, F);
    REQUIRE(sys.n_rows == 4);   // 1 + 3
    REQUIRE(sys.A.is_symmetric(1e-12));
    for (mac::la::Index r = 0; r < sys.A.rows(); ++r) {
        REQUIRE(sys.A.diag(r) > 0.0);
    }
}
