#include <functional>
#include <stdexcept>
#include <cmath>
#include <iostream>

#include <boost/multiprecision/mpfr.hpp>

#include <uncertainties/ureal.hpp>
#include <uncertainties/math.hpp>
#include <uncertainties/functions.hpp>

namespace unc = uncertainties;

using type = boost::multiprecision::mpfr_float;
using utype = unc::UReal<type>;

const type tol = 100 * unc::default_step<type>();

bool close(const type &x, const type &y,
           const type &atol=tol, const type &rtol=tol) {
    using std::abs;
    return abs(x - y) < abs(x + y) * rtol + atol;
}

bool check(const utype &x,
           const std::function<type(const type &)> &f,
           const std::function<utype(const utype &)> &uf) {
    const utype fx = uf(x);
    const utype nfx = unc::uunary<type>(f)(x);
    const bool ok = close(fx.n(), nfx.n()) and close(nfx.s(), fx.s());
    return ok;
}

bool check2(const utype &x, const utype &y,
            const std::function<type(const type &, const type &)> &f,
            const std::function<utype(const utype &, const utype &)> &uf) {
    const utype fxy = uf(x, y);
    const utype nfxy = unc::ubinary<type>(f)(x, y);
    const bool ok = close(fxy.n(), nfxy.n()) and close(nfxy.s(), fxy.s());
    return ok;
}

void print(const utype &a) {
    std::cout << a.n() << " pm " << a.s() << "\n";
}

void print(const type &a) {
    std::cout << a << "\n";
}

bool problems = false;
#define CHECK(X, F) if (not check(X, [](const type &x) { using std::F; return F(x); }, [](const utype &x) { return F(x); })) { std::cout << "error on " #F << "\n"; problems = true; }
#define CHECK2(X, Y, F) if (not check2(X, Y, [](const type &x, const type &y) { using std::F; return F(x, y); }, [](const utype &x, const utype &y) { return F(x, y); })) { std::cout << "error on " #F << "\n"; problems = true; }

int main() {
    utype x = {1.4, 0.8};
    utype y = {0.8, 0.5};
    utype z = {0.6, 0.3};
    CHECK(x, abs);
    CHECK(-x, abs);
    CHECK2(x, y, fmod);
    CHECK2(x, y, remainder);
    CHECK2(x, y, fmax);
    CHECK2(x, y, fmin);
    CHECK(x, exp);
    CHECK(x, exp2);
    CHECK(x, expm1);
    CHECK(x, log);
    CHECK(x, log10);
    CHECK(x, log2);
    CHECK(x, log1p);
    CHECK2(x, y, pow);
    CHECK(x, sqrt);
    CHECK(x, cbrt);
    CHECK2(x, y, hypot);
    CHECK(x, sin);
    CHECK(x, cos);
    CHECK(x, tan);
    CHECK(y, asin);
    CHECK(y, acos);
    CHECK(x, atan);
    CHECK2(y, z, atan2);
    CHECK2(y, -z, atan2);
    CHECK2(-y, z, atan2);
    CHECK2(-y, -z, atan2);
    CHECK(x, sinh);
    CHECK(x, cosh);
    CHECK(x, tanh);
    CHECK(x, asinh);
    CHECK(x, acosh);
    CHECK(y, atanh);
    CHECK(x, erf);
    CHECK(x, erfc);
    if (problems) {
        return 1;
    }
    return 0;
}
