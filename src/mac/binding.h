#pragma once

#include <vector>

#include "mac/mac_world.h"

namespace mac::macsim {

struct BindingOpts {
    Real r_bind        = 0.3;     // world-frame distance threshold between sites
    Real cos_threshold = 0.5;     // optional orientation check; -1 disables
    int  max_per_step  = 1000000; // hard cap for safety
};

// Scan all (body i, body j, site_i, site_j) tuples. Fires a weld constraint
// for the first eligible pair, marks both sites used, returns count fired.
//
// Reciprocity rule: site_i.partner_kind == kind(j) && site_j.partner_kind ==
// kind(i) && site_i.tag == site_j.tag.
//
// O(N^2 * S^2) where N = bodies, S = max sites/protein. Fine for N ≲ 1000;
// swap in a cell list for large N.
int detect_and_bind(MacWorld& mw, const BindingOpts& opts = {});

} // namespace mac::macsim
