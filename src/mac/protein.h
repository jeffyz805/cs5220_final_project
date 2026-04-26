#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "physics/math3d.h"

namespace mac::macsim {

using physics::Real;
using physics::Vec3;

// Stylized coarse-grained model of complement-pathway proteins. Each protein
// is a single rigid body with a small set of named binding sites in body
// frame. NOT biologically validated — the binding chemistry and geometry are
// chosen to drive a growing-constraint workload that mirrors MAC assembly.

enum class Kind : std::uint8_t { C5b = 0, C6 = 1, C7 = 2, C8 = 3, C9 = 4, NKinds = 5 };

inline std::string_view kind_name(Kind k) {
    switch (k) {
        case Kind::C5b: return "C5b";
        case Kind::C6:  return "C6";
        case Kind::C7:  return "C7";
        case Kind::C8:  return "C8";
        case Kind::C9:  return "C9";
        default: return "?";
    }
}

// A binding site lives on one protein and accepts a partner of a specific
// kind. Two sites can fuse only if their (kind, partner_kind) pairs match
// reciprocally and their world-frame attach points are within `r_bind`.
struct BindingSite {
    Vec3 r_body{0, 0, 0};
    Kind partner_kind = Kind::C5b;
    int  tag = 0;   // disambiguates multiple sites of same partner kind
};

struct ProteinSpec {
    Kind  kind;
    Real  radius  = 1.0;     // for soft repulsion / cell list
    Real  gamma_t = 1.0;
    Real  gamma_r = 1.0;
    std::vector<BindingSite> sites;
};

// Default coarse-grained MAC cascade: the canonical sequence is
//   C5b → +C6 → +C7 → +C8 → +(many C9)
// Each protein's binding sites are arranged to cascade in this order.
//
// Geometry: spheres of radius 1.0; binding sites placed on opposite poles so
// growing chain is roughly linear, mimicking a transmembrane pore.
std::vector<ProteinSpec> default_specs();

} // namespace mac::macsim
