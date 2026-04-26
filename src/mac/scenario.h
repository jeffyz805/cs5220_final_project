#pragma once

#include <cstdint>
#include <random>

#include "mac/mac_world.h"

namespace mac::macsim {

struct ScenarioOpts {
    // Counts per kind. Default: 5 of each.
    int n_C5b = 5, n_C6 = 5, n_C7 = 5, n_C8 = 5, n_C9 = 5;

    Real box_size = 8.0;          // initial placement cube ±box_size
    Real min_sep  = 2.5;          // min initial center-to-center distance

    Real kT       = 1.0;
    Real dt       = 5e-3;
    Real eps_reg  = 1e-6;
    Real baumgarte = 5.0;

    // Soft confining harmonic well: F_lin = -k_conf * x. Keeps bodies bounded
    // without periodic wrap. Set 0 to disable.
    Real k_conf = 0.5;

    std::uint64_t seed = 0xCA5EEDULL;
};

// Build a fresh MacWorld with proteins placed at random in the initial box,
// with rejection sampling for the minimum separation. Sites all start unused.
MacWorld make_scenario(const ScenarioOpts& opts = {});

// Apply soft confining potential into F_body (size 6 * n_bodies).
// Linear: F = -k_conf * x. Angular: zero.
void apply_confining_force(const MacWorld& mw, Real k_conf,
                           std::span<Real> F_body);

} // namespace mac::macsim
