#pragma once

#include <span>
#include <vector>

#include "constraints/constraint.h"
#include "la/sparse_matrix.h"
#include "physics/rigid_body.h"

namespace mac::constraints {

// Assembled linear system for a single constraint solve.
//
// Layout:
//   - Each constraint i occupies dof(i) consecutive rows in A. The mapping
//     constraint_index -> first_row is `con_row_start`.
//   - n_rows = sum of dof(c) over c
//   - Each body has 6 generalized coordinates [v_x v_y v_z ω_x ω_y ω_z]
//     contiguous in the body-space vectors (size 6 * n_bodies).
//
// We do not materialize J as a sparse matrix. Instead we build A = J M^-1 J^T
// directly by iterating constraint-pair × shared-body, plus add ε I.
struct ConstraintSystem {
    la::CsrMatrix A;                    // n_rows x n_rows, SPD
    std::vector<la::Real> rhs;          // length n_rows; -β/Δt · C - J M^-1 F
    std::vector<int>      con_row_start; // size constraints.size() + 1
    int n_rows = 0;

    // Cached per-constraint Jacobian blocks (row-major dof×6 each), so we
    // can apply J^T λ later without recomputing.
    std::vector<std::vector<la::Real>> Ja_blocks;  // size = constraints.size()
    std::vector<std::vector<la::Real>> Jb_blocks;
};

// Build the constraint system at the current world state with current per-body
// generalized force `F_body` (size 6 * n_bodies).
ConstraintSystem assemble(const physics::World& world,
                          const std::vector<Constraint>& cons,
                          std::span<const la::Real> F_body);

// Compute correction force ΔF = J^T λ, accumulated into F_body (size 6*n_bodies).
void apply_jt_lambda(const physics::World& world,
                     const std::vector<Constraint>& cons,
                     const ConstraintSystem& sys,
                     std::span<const la::Real> lambda,
                     std::span<la::Real> F_body);

// Per-body inverse mobility times a body-space vector v (size 6 * n_bodies).
// Since mobility is isotropic per body: out[i] = (1/γ_t for linear,
// 1/γ_r for angular) component-wise.
void apply_inv_mobility(const physics::World& world,
                        std::span<const la::Real> v,
                        std::span<la::Real> out);

} // namespace mac::constraints
