#include <iostream>
#include <cassert>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

int main() {
    unc::udouble2e x = unc::distr::normal<unc::udouble2e>(3, 2);
    assert(x.s() == 2);
    assert(x.n() == 3);
    assert(x.format() == "3.0 ± 2.0");
}
