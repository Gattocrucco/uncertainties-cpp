#include <cmath>
#include <functional>
#include <stdexcept>

#include <uncertainties/ureal.hpp>
#include <uncertainties/math.hpp>
#include <uncertainties/functions.hpp>

bool close(double x, double y, double atol=1e-6, double rtol=1e-6) {
    return std::abs(x - y) < std::abs(x + y) * rtol + atol;
}

namespace unc = uncertainties;
using unc::udouble;

bool check(const udouble &x,
           const std::function<double(double)> &f,
           const std::function<udouble(const udouble &)> &uf) {
    const udouble fx = uf(x);
    const udouble nfx = unc::uunary<double>(f)(x);
    const bool ok = close(fx.n(), nfx.n()) and close(nfx.s(), fx.s());
    return ok;
}

bool check2(const udouble &x, const udouble &y,
            const std::function<double(double, double)> &f,
            const std::function<udouble(const udouble &, const udouble &)> &uf) {
    const udouble fxy = uf(x, y);
    const udouble nfxy = unc::ubinary<double>(f)(x, y);
    const bool ok = close(fxy.n(), nfxy.n()) and close(nfxy.s(), fxy.s());
    return ok;
}

#define CHECK(X, F) if (not check(X, [](const double x) { return std::F(x); }, [](const udouble &x) { return unc::F(x); })) { throw std::runtime_error("error on " #F); }
#define CHECK2(X, Y, F) if (not check2(X, Y, [](const double x, const double y) { return std::F(x, y); }, [](const udouble &x, const udouble &y) { return unc::F(x, y); })) { throw std::runtime_error("error on " #F); }

int main() {
    udouble x = {1.4, 0.8};
    udouble y = {0.8, 0.5};
    udouble z = {0.6, 0.3};
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
}
