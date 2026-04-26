#pragma once

#include <array>
#include <span>
#include <vector>

#include "physics/rigid_body.h"

namespace mac::constraints {

using physics::RigidBody;
using physics::Vec3;
using physics::Real;

// Each constraint contributes c_dof rows to the global Jacobian J. Those rows
// touch (at most) two bodies; for each touched body the constraint emits a
// dense (c_dof × 6) Jacobian block and the constraint violation C(x) (length
// c_dof). The sign convention is C(x) = 0 when the constraint is satisfied,
// and the Baumgarte bias is -β/Δt · C(x).

enum class ConstraintKind : std::uint8_t { Distance = 1, BallJoint = 3, Weld = 6 };

// 6-wide Jacobian block per body: layout = [d/dx_lin (3), d/dx_ang (3)].
using JacBlock = std::array<Real, 6>;     // single row of 6 entries (one DoF)

struct Constraint {
    ConstraintKind kind;

    // Body indices into World.bodies. body_b == -1 means anchored to world frame
    // (one-sided constraint); body_b only used otherwise.
    int  body_a = -1;
    int  body_b = -1;

    // Body-frame attach points (local offsets from body COM).
    Vec3 r_a = {0, 0, 0};
    Vec3 r_b = {0, 0, 0};

    // Distance constraint target length (only used if kind == Distance).
    Real target_dist = 0.0;

    // For Weld: target relative orientation = q_a^-1 * q_b at bind time.
    // Stored as the "rest" quaternion. Only used if kind == Weld.
    physics::Quat rest_orient = physics::Quat::identity();

    int dof() const { return static_cast<int>(kind); }
};

// Compute constraint violation C(x) into `out` (size dof()).
void compute_violation(const Constraint& c, const RigidBody& A,
                       const RigidBody* B, std::span<Real> out);

// Compute Jacobian rows for this constraint. `Ja` and `Jb` are filled with
// dof() rows × 6 cols, row-major (so size dof()*6). If body_b == -1, Jb is
// untouched.
void compute_jacobian(const Constraint& c, const RigidBody& A,
                      const RigidBody* B,
                      std::span<Real> Ja, std::span<Real> Jb);

// Helpers (also used in the assembler).
namespace detail {

// Distance constraint: C(x) = ||p_a - p_b|| - L.
void distance_violation(const Constraint& c, const RigidBody& A,
                        const RigidBody& B, std::span<Real> out);
void distance_jacobian (const Constraint& c, const RigidBody& A,
                        const RigidBody& B,
                        std::span<Real> Ja, std::span<Real> Jb);

// Ball-joint: C(x) = p_a - p_b (3-vector).
void balljoint_violation(const Constraint& c, const RigidBody& A,
                         const RigidBody& B, std::span<Real> out);
void balljoint_jacobian (const Constraint& c, const RigidBody& A,
                         const RigidBody& B,
                         std::span<Real> Ja, std::span<Real> Jb);

// Weld (6-DOF): linear part = ball-joint, angular part = relative orientation
// error. The angular error is the imaginary part of (q_a^-1 * q_b * rest^-1)
// scaled by 2 (small-angle approximation); see Baraff-Witkin / GDC notes.
void weld_violation(const Constraint& c, const RigidBody& A,
                    const RigidBody& B, std::span<Real> out);
void weld_jacobian (const Constraint& c, const RigidBody& A,
                    const RigidBody& B,
                    std::span<Real> Ja, std::span<Real> Jb);

} // namespace detail

} // namespace mac::constraints
