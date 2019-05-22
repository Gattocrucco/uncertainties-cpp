#include <iostream>
#include <cassert>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

int main() {
    const unc::udouble2e x = unc::distr::normal<unc::udouble2e>(3, 2);
    const unc::udouble2e y = -x;
}
