// MAC scenario tests.
//   1. Scenario builder produces requested body count + spec consistency.
//   2. Forced-proximity placement triggers a binding event.
//   3. Constraint count grows monotonically over the run.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <random>

#include "mac/binding.h"
#include "mac/mac_world.h"
#include "mac/protein.h"
#include "mac/scenario.h"
#include "physics/integrator.h"

using namespace mac::macsim;

TEST_CASE("scenario count and metadata consistency", "[mac][scenario]") {
    ScenarioOpts o;
    o.n_C5b = 2; o.n_C6 = 3; o.n_C7 = 1; o.n_C8 = 1; o.n_C9 = 4;
    auto mw = make_scenario(o);
    REQUIRE(mw.world.n_bodies() == 11);
    REQUIRE(static_cast<int>(mw.body_to_kind.size()) == 11);
    REQUIRE(mw.site_used.size() == 11);
    for (int i = 0; i < mw.world.n_bodies(); ++i) {
        REQUIRE(mw.site_used[i].size() == mw.spec_of(i).sites.size());
    }
}

TEST_CASE("forced-proximity C5b/C6 fires a binding", "[mac][binding]") {
    MacWorld mw;
    mw.specs = default_specs();
    // Place C5b and C6 with their +X / -X sites kissing.
    mac::physics::RigidBody A, B;
    A.x = {0, 0, 0};
    B.x = {2.0, 0, 0};   // sites at A+X (1,0,0) and B-X (1,0,0) coincide
    A.gamma_t = A.gamma_r = 1.0;
    B.gamma_t = B.gamma_r = 1.0;
    mw.world.bodies = {A, B};
    mw.body_to_kind = {static_cast<int>(Kind::C5b), static_cast<int>(Kind::C6)};
    mw.site_used.assign(2, std::vector<bool>{});
    mw.site_used[0].assign(mw.specs[0].sites.size(), false);
    mw.site_used[1].assign(mw.specs[1].sites.size(), false);

    BindingOpts bopts;
    bopts.r_bind = 0.5;
    int fired = detect_and_bind(mw, bopts);
    REQUIRE(fired == 1);
    REQUIRE(mw.constraints.size() == 1);
    REQUIRE(mw.constraints[0].kind == mac::constraints::ConstraintKind::Weld);
    REQUIRE(mw.constraints[0].body_a == 0);
    REQUIRE(mw.constraints[0].body_b == 1);
}

TEST_CASE("constraint count is monotone non-decreasing across run",
          "[mac][monotonicity]") {
    ScenarioOpts so;
    so.seed = 42;
    so.box_size = 6.0; so.min_sep = 2.0;
    auto mw = make_scenario(so);

    BindingOpts bopts; bopts.r_bind = 0.4;

    mac::physics::IntegratorOpts iopts;
    iopts.solver = mac::physics::IntegratorOpts::Solver::PCG;
    iopts.iter_opts.rtol = 1e-8;
    iopts.iter_opts.max_iter = 500;
    iopts.seed = 7;

    std::mt19937_64 rng(iopts.seed);

    std::size_t prev = 0;
    for (int s = 0; s < 1500; ++s) {
        detect_and_bind(mw, bopts);
        REQUIRE(mw.constraints.size() >= prev);
        prev = mw.constraints.size();
        auto fext = [&](const mac::physics::World&, std::span<mac::la::Real> F){
            apply_confining_force(mw, /*k_conf=*/0.5, F);
        };
        mac::physics::step(mw.world, mw.constraints, fext, rng, iopts);
    }
    // At least *some* binding should have happened with 25 bodies in 1500 steps.
    REQUIRE(mw.n_binding_events_total > 0);
}
