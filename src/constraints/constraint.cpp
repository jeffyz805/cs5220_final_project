#include "constraints/constraint.h"

#include <cassert>
#include <cmath>

namespace mac::constraints {

using physics::Quat;

namespace {

// Helpers for writing rows (c_dof x 6) of a Jacobian block in row-major.
struct Row6 { Real* p; };
inline void set_row6(Row6 r, Real lx, Real ly, Real lz,
                     Real ax, Real ay, Real az) {
    r.p[0] = lx; r.p[1] = ly; r.p[2] = lz;
    r.p[3] = ax; r.p[4] = ay; r.p[5] = az;
}

// World-frame attach point: p = x + R(q) * r_body.
inline Vec3 attach_world(const RigidBody& B, const Vec3& r_body) {
    return B.x + B.q.rotate(r_body);
}

inline Vec3 r_world(const RigidBody& B, const Vec3& r_body) {
    return B.q.rotate(r_body);
}

} // anon

namespace detail {

// ---- Distance ----
void distance_violation(const Constraint& c, const RigidBody& A,
                        const RigidBody& B, std::span<Real> out) {
    Vec3 pa = attach_world(A, c.r_a);
    Vec3 pb = attach_world(B, c.r_b);
    Vec3 d  = pa - pb;
    out[0]  = d.norm() - c.target_dist;
}

void distance_jacobian(const Constraint& c, const RigidBody& A,
                       const RigidBody& B,
                       std::span<Real> Ja, std::span<Real> Jb) {
    Vec3 pa = attach_world(A, c.r_a);
    Vec3 pb = attach_world(B, c.r_b);
    Vec3 d  = pa - pb;
    Real n  = d.norm();
    if (n < 1e-12) n = 1e-12;
    Vec3 dh = d * (1.0 / n);

    Vec3 ra_w = r_world(A, c.r_a);
    Vec3 rb_w = r_world(B, c.r_b);
    Vec3 ra_x_dh = ra_w.cross(dh);
    Vec3 rb_x_dh = rb_w.cross(dh);

    set_row6({Ja.data()},  dh.x,  dh.y,  dh.z,  ra_x_dh.x,  ra_x_dh.y,  ra_x_dh.z);
    set_row6({Jb.data()}, -dh.x, -dh.y, -dh.z, -rb_x_dh.x, -rb_x_dh.y, -rb_x_dh.z);
}

// ---- Ball-joint (3 DoF) ----
void balljoint_violation(const Constraint& c, const RigidBody& A,
                         const RigidBody& B, std::span<Real> out) {
    Vec3 pa = attach_world(A, c.r_a);
    Vec3 pb = attach_world(B, c.r_b);
    Vec3 d  = pa - pb;
    out[0] = d.x; out[1] = d.y; out[2] = d.z;
}

// J_a (3x6) = [ I_3 | -[r_a]_x ];  J_b (3x6) = [-I_3 |  [r_b]_x ]
static void write_balljoint_blocks(const Constraint& c,
                                   const RigidBody& A, const RigidBody& B,
                                   std::span<Real> Ja, std::span<Real> Jb,
                                   int row_offset = 0) {
    Vec3 ra = r_world(A, c.r_a);
    Vec3 rb = r_world(B, c.r_b);
    // Row 0: dC0/d... = (1, 0, 0, 0, ra.z, -ra.y) for A; negate + use rb for B.
    auto* a = Ja.data() + row_offset * 6;
    auto* b = Jb.data() + row_offset * 6;
    set_row6({a + 0 * 6}, 1, 0, 0,    0,  ra.z, -ra.y);
    set_row6({a + 1 * 6}, 0, 1, 0, -ra.z,    0,  ra.x);
    set_row6({a + 2 * 6}, 0, 0, 1,  ra.y, -ra.x,   0);
    set_row6({b + 0 * 6}, -1, 0, 0,    0, -rb.z,  rb.y);
    set_row6({b + 1 * 6}, 0, -1, 0,  rb.z,    0, -rb.x);
    set_row6({b + 2 * 6}, 0, 0, -1, -rb.y,  rb.x,    0);
}

void balljoint_jacobian(const Constraint& c, const RigidBody& A,
                        const RigidBody& B,
                        std::span<Real> Ja, std::span<Real> Jb) {
    write_balljoint_blocks(c, A, B, Ja, Jb, 0);
}

// ---- Weld (6 DoF): ball-joint linear + angular orientation error ----
//
// Angular error: q_err = q_a^-1 * q_b * rest^-1, kept near identity. The
// imaginary part of q_err (small-angle vector) ≈ 0.5 * relative angular
// rotation. We use 2 * imag(q_err) so the violation maps cleanly to angular
// velocity through J = [0 | -I, 0 | I] (in body-A frame). For simplicity we
// express in world frame: angular Jacobian rows are just ±I_3 in the angular
// columns.
void weld_violation(const Constraint& c, const RigidBody& A,
                    const RigidBody& B, std::span<Real> out) {
    // Linear (rows 0..2)
    balljoint_violation(c, A, B, out.subspan(0, 3));

    // Angular: q_err = q_b * rest^-1 * q_a^-1, then 2 * imag
    Quat qa_inv = {A.q.w, -A.q.x, -A.q.y, -A.q.z};
    Quat rest_inv = {c.rest_orient.w, -c.rest_orient.x,
                     -c.rest_orient.y, -c.rest_orient.z};
    Quat q_err = B.q * rest_inv * qa_inv;
    if (q_err.w < 0.0) {  // shortest path
        q_err.w = -q_err.w; q_err.x = -q_err.x; q_err.y = -q_err.y; q_err.z = -q_err.z;
    }
    out[3] = 2.0 * q_err.x;
    out[4] = 2.0 * q_err.y;
    out[5] = 2.0 * q_err.z;
}

void weld_jacobian(const Constraint& c, const RigidBody& A,
                   const RigidBody& B,
                   std::span<Real> Ja, std::span<Real> Jb) {
    // Linear rows: ball-joint
    write_balljoint_blocks(c, A, B, Ja, Jb, 0);

    // Angular rows 3..5: J_a = [0 | -I_3],  J_b = [0 | I_3]
    auto* a = Ja.data();
    auto* b = Jb.data();
    set_row6({a + 3 * 6}, 0, 0, 0, -1, 0, 0);
    set_row6({a + 4 * 6}, 0, 0, 0,  0, -1, 0);
    set_row6({a + 5 * 6}, 0, 0, 0,  0,  0, -1);
    set_row6({b + 3 * 6}, 0, 0, 0,  1, 0, 0);
    set_row6({b + 4 * 6}, 0, 0, 0,  0, 1, 0);
    set_row6({b + 5 * 6}, 0, 0, 0,  0, 0, 1);
}

} // namespace detail

void compute_violation(const Constraint& c, const RigidBody& A,
                       const RigidBody* B, std::span<Real> out) {
    assert(c.body_b < 0 || B != nullptr);
    // Stub-anchored case: treat B as identity-located world frame.
    static const RigidBody kWorldAnchor{};
    const RigidBody& BB = (c.body_b < 0) ? kWorldAnchor : *B;
    switch (c.kind) {
        case ConstraintKind::Distance:  detail::distance_violation (c, A, BB, out); break;
        case ConstraintKind::BallJoint: detail::balljoint_violation(c, A, BB, out); break;
        case ConstraintKind::Weld:      detail::weld_violation     (c, A, BB, out); break;
    }
}

void compute_jacobian(const Constraint& c, const RigidBody& A,
                      const RigidBody* B,
                      std::span<Real> Ja, std::span<Real> Jb) {
    assert(c.body_b < 0 || B != nullptr);
    static const RigidBody kWorldAnchor{};
    const RigidBody& BB = (c.body_b < 0) ? kWorldAnchor : *B;
    switch (c.kind) {
        case ConstraintKind::Distance:  detail::distance_jacobian (c, A, BB, Ja, Jb); break;
        case ConstraintKind::BallJoint: detail::balljoint_jacobian(c, A, BB, Ja, Jb); break;
        case ConstraintKind::Weld:      detail::weld_jacobian     (c, A, BB, Ja, Jb); break;
    }
}

} // namespace mac::constraints
