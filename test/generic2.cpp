#include <iostream>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

int main() {
    unc::udouble2e x = unc::distr::normal<unc::udouble2e>(3, 2);
    std::cout << x << "\n";
}
