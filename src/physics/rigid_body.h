#pragma once

#include <cstdint>
#include <vector>

#include "physics/math3d.h"

namespace mac::physics {

// Coarse-grained rigid body for overdamped Brownian dynamics.
//
// State:    position x (Vec3), orientation q (Quat).
// Mobility: isotropic translational drag γ_t and rotational drag γ_r.
//           inverse mobility = diag(1/γ_t * I_3, 1/γ_r * I_3) in the 6-DOF
//           generalized force ↔ velocity relation.
//
// We don't store velocity (overdamped: velocity is the instantaneous mobility
// response to the current force — not persisted between steps).
struct RigidBody {
    Vec3   x = {0, 0, 0};
    Quat   q = Quat::identity();
    Real   gamma_t = 1.0;    // translational drag (force / velocity)
    Real   gamma_r = 1.0;    // rotational drag (torque / angular velocity)
    Real   radius  = 1.0;    // for proximity / cell list (informational)
};

// World of bodies. Indices into `bodies` are stable for the duration of a
// simulation; constraint code holds these indices.
struct World {
    std::vector<RigidBody> bodies;

    Real kT     = 1.0;    // thermal energy (units consistent w/ gamma's)
    Real dt     = 1e-3;
    Real eps_reg = 1e-6;  // regularization on constraint matrix diagonal
    Real baumgarte = 0.2; // position-error feedback gain (β / Δt)

    int n_bodies() const { return static_cast<int>(bodies.size()); }
};

} // namespace mac::physics
