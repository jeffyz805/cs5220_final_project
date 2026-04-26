#pragma once

#include <vector>

#include "constraints/constraint.h"
#include "mac/protein.h"
#include "physics/rigid_body.h"

namespace mac::macsim {

// Wraps physics::World with protein metadata, binding-site bookkeeping, and
// the dynamic constraint list.
struct MacWorld {
    physics::World world;
    std::vector<ProteinSpec> specs;        // indexed by Kind
    std::vector<int>         body_to_kind; // size == world.n_bodies()

    // For each body, a parallel vector of bools: site_used[body][site_index].
    // Once a site is consumed, it can't bind again.
    std::vector<std::vector<bool>> site_used;

    // Active equality constraints (welds inserted by binding events).
    std::vector<constraints::Constraint> constraints;

    // Stats for logging.
    int n_binding_events_total = 0;
    int n_bindings_this_step   = 0;

    Kind kind_of(int body_idx) const {
        return static_cast<Kind>(body_to_kind[body_idx]);
    }
    const ProteinSpec& spec_of(int body_idx) const {
        return specs[body_to_kind[body_idx]];
    }
};

} // namespace mac::macsim
