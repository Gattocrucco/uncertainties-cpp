#include <boost/multiprecision/mpfr.hpp>

#include <uncertainties/ureal.hpp>
#include <uncertainties/math.hpp>

namespace unc = uncertainties;
namespace mp = boost::multiprecision;

using ump = unc::UReal<mp::mpfr_float>;

int main() {
    mp::mpfr_float a = 1;
    ump x(0.5, 1);
    ump y(0.4, 1);
    ump z;
    z = abs(x);
    z = fmod(x, y);
    z = remainder(x, y);
    z = fmax(x, y);
    z = fmin(x, y);
    z = exp(x);
    z = exp2(x);
    z = expm1(x);
    z = log(x);
    z = log10(x);
    z = log2(x);
    z = log1p(x);
    z = pow(x, y);
    z = sqrt(x);
    z = cbrt(x);
    z = sin(x);
    z = cos(x);
    z = atan(x);
    z = asin(x);
    z = acos(x);
    z = atan(x);
    z = atan2(x, y);
    z = sinh(x);
    z = cosh(x);
    z = tanh(x);
    z = asinh(x);
    z = acosh(x);
    z = atanh(x);
    z = erf(x);
    z = erfc(x);
    isfinite(x);
    isnormal(x);
}
