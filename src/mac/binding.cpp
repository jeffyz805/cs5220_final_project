#include "mac/binding.h"

#include "physics/math3d.h"

namespace mac::macsim {

using physics::Quat;
using physics::Vec3;

namespace {

// World-frame attach point of site s on body b.
Vec3 site_world(const physics::RigidBody& B, const BindingSite& s) {
    return B.x + B.q.rotate(s.r_body);
}

bool sites_compatible(Kind kind_i, const BindingSite& si,
                      Kind kind_j, const BindingSite& sj) {
    return si.partner_kind == kind_j
        && sj.partner_kind == kind_i
        && si.tag == sj.tag;
}

} // anon

int detect_and_bind(MacWorld& mw, const BindingOpts& opts) {
    auto& W = mw.world;
    const int n = W.n_bodies();
    int fired = 0;

    Real r2 = opts.r_bind * opts.r_bind;

    for (int i = 0; i < n && fired < opts.max_per_step; ++i) {
        Kind ki = mw.kind_of(i);
        const auto& Bi = W.bodies[i];
        const auto& spi = mw.specs[static_cast<int>(ki)];

        for (int j = i + 1; j < n && fired < opts.max_per_step; ++j) {
            Kind kj = mw.kind_of(j);
            const auto& Bj = W.bodies[j];
            const auto& spj = mw.specs[static_cast<int>(kj)];

            // Cheap pre-filter: COM distance.
            Vec3 d = Bi.x - Bj.x;
            Real cd2 = d.norm2();
            Real maxr = (spi.radius + spj.radius + opts.r_bind);
            if (cd2 > maxr * maxr) continue;

            for (std::size_t a = 0; a < spi.sites.size(); ++a) {
                if (mw.site_used[i][a]) continue;
                const auto& si = spi.sites[a];

                for (std::size_t b = 0; b < spj.sites.size(); ++b) {
                    if (mw.site_used[j][b]) continue;
                    const auto& sj = spj.sites[b];
                    if (!sites_compatible(ki, si, kj, sj)) continue;

                    Vec3 pa = site_world(Bi, si);
                    Vec3 pb = site_world(Bj, sj);
                    Vec3 dd = pa - pb;
                    if (dd.norm2() > r2) continue;

                    // Insert weld constraint between body i (site a) and j (site b).
                    constraints::Constraint c;
                    c.kind   = constraints::ConstraintKind::Weld;
                    c.body_a = i; c.body_b = j;
                    c.r_a    = si.r_body;
                    c.r_b    = sj.r_body;
                    // rest_orient = q_a^-1 * q_b chosen so that the violation
                    // q_err = q_b * rest^-1 * q_a^-1 is identity at bind time.
                    Quat qa_inv{Bi.q.w, -Bi.q.x, -Bi.q.y, -Bi.q.z};
                    c.rest_orient = (qa_inv * Bj.q).normalized();

                    mw.constraints.push_back(c);
                    mw.site_used[i][a] = true;
                    mw.site_used[j][b] = true;
                    ++fired;
                    break;  // body i,site a consumed; move on
                }
                if (mw.site_used[i][a]) break;
            }
        }
    }

    mw.n_bindings_this_step = fired;
    mw.n_binding_events_total += fired;
    return fired;
}

} // namespace mac::macsim
