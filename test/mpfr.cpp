#include <iostream>

#include <boost/multiprecision/mpfr.hpp>

#include <uncertainties/ureal.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;
namespace mp = boost::multiprecision;

int main() {
    unc::UReal<mp::mpfr_float> x(1, 1);
    unc::UReal<mp::mpfr_float> y(1, 1);
    std::cout << x << "\n";
    std::cout << y << "\n";
}
