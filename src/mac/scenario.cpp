#include "mac/scenario.h"

#include <cmath>

namespace mac::macsim {

using physics::RigidBody;
using physics::Vec3;
using physics::Quat;

namespace {

bool any_too_close(const std::vector<RigidBody>& bodies, const Vec3& p, Real min_sep) {
    Real m2 = min_sep * min_sep;
    for (const auto& B : bodies) {
        Vec3 d = B.x - p;
        if (d.norm2() < m2) return true;
    }
    return false;
}

Quat random_quat(std::mt19937_64& rng) {
    // Marsaglia uniform random rotation.
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    Real u1 = u(rng), u2 = u(rng), u3 = u(rng);
    Real s1 = std::sqrt(1 - u1), s2 = std::sqrt(u1);
    Real t1 = 2 * M_PI * u2, t2 = 2 * M_PI * u3;
    return Quat{s2 * std::cos(t2), s1 * std::sin(t1),
                s1 * std::cos(t1), s2 * std::sin(t2)};
}

} // anon

MacWorld make_scenario(const ScenarioOpts& opts) {
    MacWorld mw;
    mw.specs = default_specs();

    mw.world.kT        = opts.kT;
    mw.world.dt        = opts.dt;
    mw.world.eps_reg   = opts.eps_reg;
    mw.world.baumgarte = opts.baumgarte;

    std::mt19937_64 rng(opts.seed);
    std::uniform_real_distribution<Real> ud(-opts.box_size, opts.box_size);

    auto place = [&](Kind k, int count) {
        const auto& spec = mw.specs[static_cast<int>(k)];
        for (int i = 0; i < count; ++i) {
            Vec3 p;
            for (int t = 0; t < 1000; ++t) {
                p = {ud(rng), ud(rng), ud(rng)};
                if (!any_too_close(mw.world.bodies, p, opts.min_sep)) break;
            }
            RigidBody B;
            B.x = p;
            B.q = random_quat(rng);
            B.gamma_t = spec.gamma_t;
            B.gamma_r = spec.gamma_r;
            B.radius  = spec.radius;
            mw.world.bodies.push_back(B);
            mw.body_to_kind.push_back(static_cast<int>(k));
            mw.site_used.emplace_back(spec.sites.size(), false);
        }
    };

    place(Kind::C5b, opts.n_C5b);
    place(Kind::C6,  opts.n_C6);
    place(Kind::C7,  opts.n_C7);
    place(Kind::C8,  opts.n_C8);
    place(Kind::C9,  opts.n_C9);

    return mw;
}

void apply_confining_force(const MacWorld& mw, Real k_conf, std::span<Real> F_body) {
    if (k_conf <= 0.0) return;
    for (int b = 0; b < mw.world.n_bodies(); ++b) {
        const auto& B = mw.world.bodies[b];
        F_body[6 * b + 0] -= k_conf * B.x.x;
        F_body[6 * b + 1] -= k_conf * B.x.y;
        F_body[6 * b + 2] -= k_conf * B.x.z;
    }
}

} // namespace mac::macsim
