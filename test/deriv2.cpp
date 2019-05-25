#include <cmath>
#include <vector>
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
void checkgrad(const unc::UReal2<Real, prop> &f,
               const unc::UReal2<Real, prop> &x,
               const Real &g) {
    const Real grad = f._grad(x);
    if (not close(g, grad)) {
        std::ostringstream s;
        s << "expected df/dx = " << g << ", got " << grad;
        s << ", with f = " << f << ", x = " << x;
        throw std::runtime_error(s.str());
    }
}

template<typename Real, unc::Prop prop>
void checkhess(const unc::UReal2<Real, prop> &f,
               const unc::UReal2<Real, prop> &x,
               const unc::UReal2<Real, prop> &y,
               const Real &h) {
    const Real hess = f._hess(x, y);
    if (not close(h, hess)) {
        std::ostringstream s;
        s << "expected ddf/dxdy = " << h << ", got " << hess;
        s << ", with f = " << f << ", x = " << x << ", y = " << y;
        throw std::runtime_error(s.str());
    }
}

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &f,
           const unc::UReal2<Real, prop> &x,
           const Real &g, const Real &h) {
    checkgrad(f, x, g);
    checkhess(f, x, x, h);
}

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &f,
           const unc::UReal2<Real, prop> &x,
           const unc::UReal2<Real, prop> &y,
           const Real &gx, const Real &gy,
           const Real &hxx, const Real &hyy, const Real &hxy) {
    checkgrad(f, x, gx);
    checkgrad(f, y, gy);
    checkhess(f, x, x, hxx);
    checkhess(f, y, y, hyy);
    checkhess(f, x, y, hxy);
}

int main() {
    using utype = unc::udouble2e;
    std::vector<utype> x;
    for (int i = 0; i < 10; ++i) {
        x.push_back(unc::distr::normal<utype>(i + 1, 1)); // keep s = 1
    }
    std::vector<utype::real_type> xn;
    for (const utype &u : x) {
        xn.push_back(u.first_order_n());
    }
    
    check(x[0], x[0], 1.0, 0.0);
    check(x[1], x[0], 0.0, 0.0);
    
    check(+x[0], x[0], 1.0, 0.0);
    check(+x[0], x[1], 0.0, 0.0);
    
    check(-x[0], x[0], -1.0, 0.0);
    check(-x[0], x[1], 0.0, 0.0);
    check(-x[0], -x[0], 1.0, 0.0);
    check(x[0], 2 * x[0], 0.5, 0.0);
    
    auto y = x[0] + x[1];
    check(y, x[0], x[1], 1.0, 1.0, 0.0, 0.0, 0.0);
    check(y, x[2], 0.0, 0.0);
    
    y = x[0] - x[1];
    check(y, x[0], x[1], 1.0, -1.0, 0.0, 0.0, 0.0);
    check(y, x[2], 0.0, 0.0);

    y = x[0] * x[1];
    check(y, x[0], x[1], xn[1], xn[0], 0.0, 0.0, 1.0);
    check(y, x[2], 0.0, 0.0);

    y = x[0] / x[1];
    check(y, x[0], x[1], 1 / xn[1], -xn[0] / (xn[1] * xn[1]), 0.0, 2 * xn[0] / (xn[1] * xn[1] * xn[1]), -1 / (xn[1] * xn[1]));
    check(y, x[2], 0.0, 0.0);
    
    y = x[0] + x[1] * x[2] / -x[3];
    check(y, x[0], 1.0, 0.0);
    check(y, x[1], xn[2] / -xn[3], 0.0);
    check(y, x[2], xn[1] / -xn[3], 0.0);
    check(y, x[3], xn[1] * xn[2] / (xn[3] * xn[3]), -2 * xn[1] * xn[2] / (xn[3] * xn[3] * xn[3]));
    check(y, x[4], 0.0, 0.0);
    checkhess(y, x[0], x[1], 0.0);
    checkhess(y, x[0], x[2], 0.0);
    checkhess(y, x[0], x[3], 0.0);
    checkhess(y, x[1], x[2], 1 / -xn[3]);
    checkhess(y, x[1], x[3], xn[2] / (xn[3] * xn[3]));
    checkhess(y, x[2], x[3], xn[1] / (xn[3] * xn[3]));
    
    y = x[0] + -x[1] * x[1] / x[0];
    check(y, -x[0], -1.0 - xn[1] * xn[1] / (xn[0] * xn[0]), -2 * xn[1] * xn[1] / (xn[0] * xn[0] * xn[0]));
    check(y, x[1], -2 * xn[1] / xn[0], -2 / xn[0]);
}
