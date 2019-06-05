#include <stdexcept>
#include <sstream>
#include <initializer_list>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

#include <ios>
#include <iomanip>

namespace unc = uncertainties;

template<typename Real>
bool close(const Real &x, const Real &y, const Real &atol=1e-8, const Real &rtol=1e-8) {
    using std::abs;
    return abs(x - y) <= atol + (abs(x) + abs(y)) * rtol;
}

template<typename Real, unc::Prop prop>
void checkcov(const unc::UReal2<Real, prop> &x, const unc::UReal2<Real, prop> &y, const Real &c) {
    const Real cov1 = cov(x, y);
    const Real cov2 = cov(y, x);
    if (not close(c, cov1)) {
        std::ostringstream s;
        s << "expected cov(x,y) = " << c << ", got " << cov1;
        s << ", with x = " << x << ", y = " << y;
        throw std::runtime_error(s.str());
    }
    if (not close(cov1, cov2)) {
        std::ostringstream s;
        s << "cov(x,y) = " << cov1 << " but cov(y,x) = " << cov2;
        s << ", with x = " << x << ", y = " << y;
        throw std::runtime_error(s.str());
    }
}

template<typename Real, unc::Prop prop>
void checkmn(const unc::UReal2<Real, prop> &x, const Real &mn, const int n) {
    Real m = x.m(n);
    if (n == 1) {
        m += x.first_order_n();
    }
    if (not close(mn, m)) {
        std::ostringstream s;
        s << "expected m(" << n << ") = " << mn << ", got " << m;
        s << ", with x = " << x;
        throw std::runtime_error(s.str());
    }
    if (n == 2) {
        const Real c = cov(x, x);
        if (not close(c, m)) {
            std::ostringstream s;
            s << std::setprecision(15) << std::scientific;
            s << "var(x) = " << m << " but cov(x,x) = " << c;
            s << ", with x = " << x;
            throw std::runtime_error(s.str());
        }
    }
}

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &x, const std::initializer_list<Real> &mlist) {
    int i = 0;
    for (const auto &m : mlist) {
        checkmn(x, m, ++i);
    }
}

using namespace unc::distr;
using utype = unc::udouble2e;

int main() {
    check(normal<utype>(1, 1), {1.0, 1.0, 0.0});
    check(normal<utype>(1, 2), {1.0, 4.0, 0.0});
    
    // sum of normal is normal
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            const utype::real_type v = i * i + j * j;
            check(normal<utype>(0, i) + normal<utype>(0, j), {0.0, v, 0.0});
        }
    }

    check(chisquare<utype>(0), {0.0, 0.0, 0.0});
    check(chisquare<utype>(1), {1.0, 2.0, 8.0});
    check(chisquare<utype>(2), {2.0, 4.0, 16.0});
    
    // sum of chisquare is chisquare
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            const utype::real_type k = i + j;
            check(chisquare<utype>(i) + chisquare<utype>(j), {k, 2 * k, 8 * k});
        }
    }
    
    // sum of normal squared is chisquare
    for (int i = 0; i < 10; ++i) {
        utype x;
        for (int k = 0; k < i; ++k) {
            utype n = normal<utype>(0, 1);
            x = x + n * n;
        }
        for (int j = 0; j < 10; ++j) {
            utype y;
            for (int k = 0; k < j; ++k) {
                utype n = normal<utype>(0, 1);
                y = y + n * n;
            }
            const utype::real_type k = i + j;
            check(x + y, {k, 2 * k, 8 * k});
        }
    }
    
    check(normal<utype>(1, 1) + normal<utype>(4, 2), {5.0, 5.0, 0.0});
    utype x = normal<utype>(1, 1);
    check(x - x, {0.0, 0.0, 0.0});
    check(x + x, {2.0, 4.0, 0.0});
    check(2 * x, {2.0, 4.0, 0.0});
    check(x + 2 * x, {3.0, 9.0, 0.0});
    checkcov(x, x, 1.0);
    checkcov(x, utype(1.0), 0.0);
    checkcov(x + x, x - x, 0.0);
}
