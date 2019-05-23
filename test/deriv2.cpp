#include <cmath>
#include <stdexcept>
#include <vector>
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
void checkgrad(const unc::UReal2<Real, prop> &f, const unc::UReal2<Real, prop> &x, const Real &g) {
    const Real grad = f._grad(x);
    if (not close(g, grad)) {
        std::ostringstream s;
        s << "expected df/dx = " << g << ", got " << grad;
        s << ", with f = " << f << ", x = " << x;
        throw std::runtime_error(s.str());
    }
}

int main() {
    using utype = unc::udouble2e;
    std::vector<utype> x;
    for (int i = 0; i < 10; ++i) {
        x.push_back(unc::distr::normal<utype>(1, 1));
    }
    std::vector<utype::real_type> xn;
    for (const utype &u : x) {
        xn.push_back(u.first_order_n());
    }
    
    checkgrad(x[0], x[0], 1.0);
    checkgrad(x[1], x[0], 0.0);
    
    checkgrad(+x[0], x[0], 1.0);
    checkgrad(+x[0], x[1], 0.0);
    
    checkgrad(-x[0], x[0], -1.0);
    checkgrad(-x[0], x[1], 0.0);
    checkgrad(-x[0], -x[0], 1.0);
    
    auto y = x[0] + x[1];
    checkgrad(y, x[0], 1.0);
    checkgrad(y, x[1], 1.0);
    checkgrad(y, x[2], 0.0);
    
    y = x[0] - x[1];
    checkgrad(y, x[0], 1.0);
    checkgrad(y, x[1], -1.0);
    checkgrad(y, x[2], 0.0);

    y = x[0] * x[1];
    checkgrad(y, x[0], xn[1]);
    checkgrad(y, x[1], xn[0]);
    checkgrad(y, x[2], 0.0);

    y = x[0] / x[1];
    checkgrad(y, x[0], 1 / xn[1]);
    checkgrad(y, x[1], -xn[0] / (xn[1] * xn[1]));
    checkgrad(y, x[2], 0.0);
    
    y = x[0] + x[1] * x[2] / -x[3];
    checkgrad(y, x[0], 1.0);
    checkgrad(y, x[1], xn[2] / -xn[3]);
    checkgrad(y, x[2], xn[1] / -xn[3]);
    checkgrad(y, x[3], xn[1] * xn[2] / (xn[3] * xn[3]));
}
