#include "constraints/assembler.h"

#include <cassert>
#include <unordered_map>

namespace mac::constraints {

using la::CsrBuilder;
using la::CsrMatrix;
using la::Index;
using la::Real;

namespace {

// Per-body inverse-mobility diagonal (6 entries: lin x lin x lin | ang x ang x ang).
inline void inv_mob_diag(const physics::RigidBody& B, Real out[6]) {
    Real it = 1.0 / B.gamma_t;
    Real ir = 1.0 / B.gamma_r;
    out[0] = out[1] = out[2] = it;
    out[3] = out[4] = out[5] = ir;
}

// row contributions of constraint c (one of dof rows) - 6 entries against
// body's 6-vector: returns the dot J_row · M^-1 · J'_row over the shared body.
//
// J_block layout: dof rows × 6 cols, row-major.
inline Real row_dot_invmob(const Real* Ja_row, const Real* Jb_row, const Real diag[6]) {
    Real s = 0;
    for (int k = 0; k < 6; ++k) s += Ja_row[k] * diag[k] * Jb_row[k];
    return s;
}

} // anon

void apply_inv_mobility(const physics::World& world,
                        std::span<const Real> v,
                        std::span<Real> out) {
    assert(static_cast<int>(v.size()) == 6 * world.n_bodies());
    assert(static_cast<int>(out.size()) == 6 * world.n_bodies());
    for (int b = 0; b < world.n_bodies(); ++b) {
        Real d[6]; inv_mob_diag(world.bodies[b], d);
        for (int k = 0; k < 6; ++k) out[6 * b + k] = d[k] * v[6 * b + k];
    }
}

ConstraintSystem assemble(const physics::World& world,
                          const std::vector<Constraint>& cons,
                          std::span<const Real> F_body) {
    const int n_bodies = world.n_bodies();
    assert(static_cast<int>(F_body.size()) == 6 * n_bodies);

    ConstraintSystem sys;
    sys.con_row_start.assign(cons.size() + 1, 0);
    for (std::size_t i = 0; i < cons.size(); ++i) {
        sys.con_row_start[i + 1] = sys.con_row_start[i] + cons[i].dof();
    }
    sys.n_rows = sys.con_row_start.back();

    // Cache per-constraint Jacobian blocks.
    sys.Ja_blocks.resize(cons.size());
    sys.Jb_blocks.resize(cons.size());

    // Per-constraint contribution to RHS: bias = -β/Δt · C - J M^-1 F.
    sys.rhs.assign(sys.n_rows, 0.0);

    // For A assembly: we need to know, for each body, which constraints touch
    // it. Then for each body B we add Σ_{i,j touching B} J_i · M_B^-1 · J_j^T
    // into A[i_rows, j_rows] (sub-block of dof_i × dof_j).
    std::vector<std::vector<int>> bodies_to_cons(n_bodies);
    for (std::size_t ci = 0; ci < cons.size(); ++ci) {
        const auto& c = cons[ci];
        if (c.body_a >= 0) bodies_to_cons[c.body_a].push_back(static_cast<int>(ci));
        if (c.body_b >= 0) bodies_to_cons[c.body_b].push_back(static_cast<int>(ci));
    }

    // Compute Jacobian blocks + violation for each constraint.
    physics::RigidBody anchor{};
    for (std::size_t ci = 0; ci < cons.size(); ++ci) {
        const auto& c = cons[ci];
        int dof = c.dof();
        sys.Ja_blocks[ci].assign(dof * 6, 0.0);
        sys.Jb_blocks[ci].assign(dof * 6, 0.0);

        const auto& A = world.bodies[c.body_a];
        const physics::RigidBody* Bptr = (c.body_b < 0) ? &anchor
                                       : &world.bodies[c.body_b];
        compute_jacobian(c, A, Bptr,
                         sys.Ja_blocks[ci], sys.Jb_blocks[ci]);

        std::vector<Real> Cviol(dof);
        compute_violation(c, A, Bptr, Cviol);

        // RHS = -β/Δt · C - J · M^-1 · F. Compute the J · M^-1 · F part using
        // the Ja/Jb blocks and inv-mobility of each touched body.
        Real diag_a[6]; inv_mob_diag(A, diag_a);
        Real diag_b[6]; inv_mob_diag(*Bptr, diag_b);

        const Real* fA = F_body.data() + 6 * c.body_a;
        const Real* fB = (c.body_b < 0) ? nullptr : (F_body.data() + 6 * c.body_b);

        int row0 = sys.con_row_start[ci];
        for (int r = 0; r < dof; ++r) {
            const Real* Ja_row = &sys.Ja_blocks[ci][r * 6];
            Real jmf = 0.0;
            for (int k = 0; k < 6; ++k) jmf += Ja_row[k] * diag_a[k] * fA[k];
            if (c.body_b >= 0) {
                const Real* Jb_row = &sys.Jb_blocks[ci][r * 6];
                for (int k = 0; k < 6; ++k) jmf += Jb_row[k] * diag_b[k] * fB[k];
            }
            sys.rhs[row0 + r] = -world.baumgarte * Cviol[r] - jmf;
        }
    }

    // Build A in CSR form. For every pair (i, j) of constraints sharing a body,
    // add to A[i_rows, j_rows] the dof_i × dof_j matrix J_i · M_body^-1 · J_j^T,
    // where the sign is +1 if body is body_a in both (or body_b in both),
    // and -1 if mixed (body_a vs body_b — which would happen if the pair shares
    // a body in opposing roles).
    CsrBuilder bld(sys.n_rows, sys.n_rows);

    // Use a per-(i, j) accumulator keyed on (con_i, con_j) → small dense block.
    // For sparse constraint graphs each accumulator is at most a few entries.
    // We iterate constraints once, and for each, walk its neighbors.
    auto block_for = [&](int ci, bool a_side_i,
                         int cj, bool a_side_j, int body_idx) {
        const auto& Ci = cons[ci];
        const auto& Cj = cons[cj];
        int dofi = Ci.dof(), dofj = Cj.dof();
        Real diag[6]; inv_mob_diag(world.bodies[body_idx], diag);

        const Real* Ji_blk = a_side_i ? sys.Ja_blocks[ci].data()
                                       : sys.Jb_blocks[ci].data();
        const Real* Jj_blk = a_side_j ? sys.Ja_blocks[cj].data()
                                       : sys.Jb_blocks[cj].data();

        int row0 = sys.con_row_start[ci];
        int col0 = sys.con_row_start[cj];
        for (int ri = 0; ri < dofi; ++ri) {
            const Real* Ji_row = Ji_blk + ri * 6;
            for (int rj = 0; rj < dofj; ++rj) {
                const Real* Jj_row = Jj_blk + rj * 6;
                Real s = row_dot_invmob(Ji_row, Jj_row, diag);
                bld.push(row0 + ri, col0 + rj, s);
            }
        }
    };

    for (int b = 0; b < n_bodies; ++b) {
        const auto& touchers = bodies_to_cons[b];
        for (int i : touchers) {
            bool a_side_i = (cons[i].body_a == b);
            for (int j : touchers) {
                bool a_side_j = (cons[j].body_a == b);
                block_for(i, a_side_i, j, a_side_j, b);
            }
        }
    }

    // Add ε I (regularization). Builder dedupes/sums duplicates per row.
    for (int r = 0; r < sys.n_rows; ++r) bld.push(r, r, world.eps_reg);

    sys.A = bld.finalize();
    return sys;
}

void apply_jt_lambda(const physics::World& world,
                     const std::vector<Constraint>& cons,
                     const ConstraintSystem& sys,
                     std::span<const Real> lambda,
                     std::span<Real> F_body) {
    assert(static_cast<int>(F_body.size()) == 6 * world.n_bodies());
    assert(static_cast<int>(lambda.size()) == sys.n_rows);

    for (std::size_t ci = 0; ci < cons.size(); ++ci) {
        const auto& c = cons[ci];
        int dof = c.dof();
        int row0 = sys.con_row_start[ci];

        const Real* Ja = sys.Ja_blocks[ci].data();
        Real* fA = F_body.data() + 6 * c.body_a;
        for (int r = 0; r < dof; ++r) {
            Real lam = lambda[row0 + r];
            for (int k = 0; k < 6; ++k) fA[k] += Ja[r * 6 + k] * lam;
        }

        if (c.body_b >= 0) {
            const Real* Jb = sys.Jb_blocks[ci].data();
            Real* fB = F_body.data() + 6 * c.body_b;
            for (int r = 0; r < dof; ++r) {
                Real lam = lambda[row0 + r];
                for (int k = 0; k < 6; ++k) fB[k] += Jb[r * 6 + k] * lam;
            }
        }
    }
}

} // namespace mac::constraints
