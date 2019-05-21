#include <cassert>

#include <uncertainties/core.hpp>

namespace unc = uncertainties;

int main() {
    assert(not unc::lazyprop());
    UNCERTAINTIES_LAZYPROP(true) {
        assert(unc::lazyprop());
        UNCERTAINTIES_LAZYPROP(false) {
            assert(not unc::lazyprop());
        }
    }
    assert(not unc::lazyprop());
}
