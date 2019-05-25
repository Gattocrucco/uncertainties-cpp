#include <stdexcept>
#include <sstream>

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
void checkvar(const unc::UReal2<Real, prop> &x, const Real &var) {
    const Real v = x.m(2);
    if (not close(v, var)) {
        std::ostringstream s;
        s << "expected s^2 = " << var << ", got " << v;
        s << ", with x = " << x;
        throw std::runtime_error(s.str());
    }
}

using namespace unc::distr;
using utype = unc::udouble2e;

int main() {
    checkvar(normal<utype>(1, 1), 1.0);
    checkvar(normal<utype>(1, 1) + normal<utype>(4, 2), 5.0);
    utype x = normal<utype>(1, 1);
    checkvar(x - x, 0.0);
    checkvar(x + x, 4.0);
    checkvar(2 * x, 4.0);
    checkvar(x + 2 * x, 9.0);
}
