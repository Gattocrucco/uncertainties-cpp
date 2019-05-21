#include <cassert>
#include <thread>

#include <uncertainties/core.hpp>

namespace unc = uncertainties;

void check() {
    const bool start = unc::lazyprop();
    for (int i = 0; i < 10000; ++i) {
        UNCERTAINTIES_LAZYPROP(true) {
            assert(unc::lazyprop());
            UNCERTAINTIES_LAZYPROP(false) {
                assert(not unc::lazyprop());
            }
            assert(unc::lazyprop());
        }
        assert(start == unc::lazyprop());
    }
}

int main() {
    assert(not unc::lazyprop());
    check();
    std::thread t1(check), t2(check);
    t1.join();
    t2.join();
    check();
}
