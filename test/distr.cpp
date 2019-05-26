#include <stdexcept>
#include <sstream>
#include <initializer_list>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

template<typename Real>
bool close(const Real &x, const Real &y, const Real &atol=1e-8, const Real &rtol=1e-8) {
    using std::abs;
    return abs(x - y) <= atol + (abs(x) + abs(y)) * rtol;
}

template<typename Real, unc::Prop prop>
void checkmn(const unc::UReal2<Real, prop> &x, const Real &mn, const int n) {
    const Real m = x.m(n);
    if (not close(mn, m)) {
        std::ostringstream s;
        s << "expected m(" << n << ") = " << mn << ", got " << m;
        s << ", with x = " << x;
        throw std::runtime_error(s.str());
    }
}

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &x, const std::initializer_list<Real> &mlist) {
    int i = 1;
    for (const auto &m : mlist) {
        checkmn(x, m, ++i);
    }
}

using namespace unc::distr;
using utype = unc::udouble2e;

int main() {
    check(normal<utype>(1, 1), {1.0, 0.0});
    check(normal<utype>(1, 1) + normal<utype>(4, 2), {5.0, 0.0});
    utype x = normal<utype>(1, 1);
    check(x - x, {0.0, 0.0});
    check(x + x, {4.0, 0.0});
    check(2 * x, {4.0, 0.0});
    check(x + 2 * x, {9.0, 0.0});
}
