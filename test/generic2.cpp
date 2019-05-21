#include <iostream>
#include <cassert>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

constexpr double n = 3;
constexpr double s = 2;

template<typename Number>
void check(const Number &x) {
    assert(x.s() == 2);
    assert(x.n() == 3);
    assert(x.isindep());
    assert(x.indepid() == 1);
    assert(x.format() == "3.0 ± 2.0");
}

int main() {
    const unc::udouble2e x = unc::distr::normal<unc::udouble2e>(3, 2);
    check(x);
    const unc::udouble2m y(x);
    check(y);
    const unc::ufloat2e z(x);
    check(z);
    const unc::ufloat2m w(x);
    check(w);
    const unc::udouble2e v = z;
    check(v);
    unc::udouble2e x1 = +x;
    check(x1);
    x1 = -x1;
    std::cout << x1 << "\n";
}
