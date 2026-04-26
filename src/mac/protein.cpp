#include "mac/protein.h"

namespace mac::macsim {

std::vector<ProteinSpec> default_specs() {
    // Canonical cascade with sites on the +X face accepting "next in chain"
    // and -X face presenting "previous" (where applicable). C9 has multiple
    // copies so we let it bind to itself or to C8.
    auto site = [](Real x, Kind partner, int tag = 0) -> BindingSite {
        return {{x, 0, 0}, partner, tag};
    };

    std::vector<ProteinSpec> specs(static_cast<int>(Kind::NKinds));
    for (auto& s : specs) {
        s.radius  = 1.0;
        s.gamma_t = 1.0;
        s.gamma_r = 1.0;
    }

    // C5b: presents to C6 on +X
    specs[0].kind = Kind::C5b;
    specs[0].sites = { site( 1.0, Kind::C6) };

    // C6: accepts C5b on -X, presents to C7 on +X
    specs[1].kind = Kind::C6;
    specs[1].sites = { site(-1.0, Kind::C5b), site( 1.0, Kind::C7) };

    // C7: accepts C6 on -X, presents to C8 on +X
    specs[2].kind = Kind::C7;
    specs[2].sites = { site(-1.0, Kind::C6),  site( 1.0, Kind::C8) };

    // C8: accepts C7 on -X, presents to C9 on +X
    specs[3].kind = Kind::C8;
    specs[3].sites = { site(-1.0, Kind::C7),  site( 1.0, Kind::C9, 0) };

    // C9: accepts C8 OR another C9 on -X, presents to C9 on +X (homotypic chain)
    specs[4].kind = Kind::C9;
    specs[4].sites = { site(-1.0, Kind::C8, 0),
                       site(-1.0, Kind::C9, 1),    // homotypic accept
                       site( 1.0, Kind::C9, 1) };  // homotypic present

    return specs;
}

} // namespace mac::macsim
