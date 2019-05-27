// math.hpp
//
// Copyright (c) Giacomo Petrillo 2019
//
// This file is part of uncertainties-cpp.
//
// uncertainties-cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// uncertainties-cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with uncertainties-cpp.  If not, see <http://www.gnu.org/licenses/>.

#ifndef UNCERTAINTIES_MATH_HPP_0A89E052
#define UNCERTAINTIES_MATH_HPP_0A89E052

/*! \file
\brief Defines standard math functions on `UReal`s.

The functions are a subset of the functions from the standard header `<cmath>`.
(You can read a detailed description of them on
[cppreference](https://en.cppreference.com/w/cpp/header/cmath).)

Be careful with non smooth functions (`abs`, `fmod`, `remainder`, `fmax`,
`fmin`): if the distance of the mean from the non derivable point is comparable
with the standard deviation, probably first order (or any order) propagation
does not make sense.

*/

#include <cmath>

#include "core.hpp"

namespace uncertainties {
    template<typename Real>
    UReal<Real> abs(const UReal<Real> &x) {
        using std::abs;
        return unary(x, abs(x.n()), x.n() >= 0 ? 1 : -1);
    }
    template<typename Real>
    UReal<Real> fmod(const UReal<Real> &x, const UReal<Real> &y) {
        using std::fmod;
        return binary(x, y, fmod(x.n(), y.n()), 1, -trunc(x.n() / y.n()));
    }
    template<typename Real>
    UReal<Real> remainder(const UReal<Real> &x, const UReal<Real> &y) {
        using std::remainder;
        using std::round;
        return binary(x, y, remainder(x.n(), y.n()), 1, -round(x.n() / y.n()));
    }
    template<typename Real>
    UReal<Real> fmax(const UReal<Real> &x, const UReal<Real> &y) {
        using std::fmax;
        const Real max = fmax(x.n(), y.n());
        const bool c = max == x.n();
        return binary(x, y, max, c ? 1 : 0, c ? 0 : 1);
    }
    template<typename Real>
    UReal<Real> fmin(const UReal<Real> &x, const UReal<Real> &y) {
        using std::fmin;
        const Real min = fmin(x.n(), y.n());
        const bool c = min == x.n();
        return binary(x, y, min, c ? 1 : 0, c ? 0 : 1);
    }
    // fdim?
    template<typename Real>
    UReal<Real> exp(const UReal<Real> &x) {
        using std::exp;
        return unary(x, exp(x.n()), exp(x.n()));
    }
    template<typename Real>
    UReal<Real> exp2(const UReal<Real> &x) {
        using std::exp2;
        using std::log;
        return unary(x, exp2(x.n()), log(Real(2)) * exp2(x.n()));
    }
    template<typename Real>
    UReal<Real> expm1(const UReal<Real> &x) {
        using std::expm1;
        using std::exp;
        return unary(x, expm1(x.n()), exp(x.n()));
    }
    template<typename Real>
    UReal<Real> log(const UReal<Real> &x) {
        using std::log;
        return unary(x, log(x.n()), Real(1) / x.n());
    }
    template<typename Real>
    UReal<Real> log10(const UReal<Real> &x) {
        using std::log10;
        using std::log;
        return unary(x, log10(x.n()), Real(1) / (x.n() * log(Real(10))));
    }
    template<typename Real>
    UReal<Real> log2(const UReal<Real> &x) {
        using std::log2;
        using std::log;
        return unary(x, log2(x.n()), Real(1) / (x.n() * log(Real(2))));
    }
    template<typename Real>
    UReal<Real> log1p(const UReal<Real> &x) {
        using std::log1p;
        return unary(x, log1p(x.n()), Real(1) / (Real(1) + x.n()));
    }
    template<typename Real>
    UReal<Real> pow(const UReal<Real> &x, const UReal<Real> &y) {
        using std::pow;
        using std::log;
        const Real p = pow(x.n(), y.n());
        return binary(x, y, p, p * y.n() / x.n(), p * log(x.n()));
    }
    template<typename Real>
    UReal<Real> sqrt(const UReal<Real> &x) {
        using std::sqrt;
        return unary(x, sqrt(x.n()), Real(1) / (2 * sqrt(x.n())));
    }
    template<typename Real>
    UReal<Real> cbrt(const UReal<Real> &x) {
        using std::cbrt;
        using std::pow;
        return unary(x, cbrt(x.n()), pow(x.n(), -Real(2) / Real(3)) / Real(3));
    }
    template<typename Real>
    UReal<Real> hypot(const UReal<Real> &x, const UReal<Real> &y) {
        using std::hypot;
        const Real h = hypot(x.n(), y.n());
        return binary(x, y, h, x.n() / h, y.n() / h);
    }
    template<typename Real>
    UReal<Real> sin(const UReal<Real> &x) {
        using std::sin;
        using std::cos;
        return unary(x, sin(x.n()), cos(x.n()));
    }
    template<typename Real>
    UReal<Real> cos(const UReal<Real> &x) {
        using std::cos;
        using std::sin;
        return unary(x, cos(x.n()), -sin(x.n()));
    }
    template<typename Real>
    UReal<Real> tan(const UReal<Real> &x) {
        using std::tan;
        const Real t = tan(x.n());
        return unary(x, t, Real(1) + t * t);
    }
    template<typename Real>
    UReal<Real> asin(const UReal<Real> &x) {
        using std::asin;
        using std::sqrt;
        return unary(x, asin(x.n()), Real(1) / sqrt(1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> acos(const UReal<Real> &x) {
        using std::acos;
        using std::sqrt;
        return unary(x, acos(x.n()), -Real(1) / sqrt(1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> atan(const UReal<Real> &x) {
        using std::atan;
        return unary(x, atan(x.n()), Real(1) / (1 + x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> atan2(const UReal<Real> &x, const UReal<Real> &y) {
        using std::atan2;
        const Real yx = y.n() / x.n();
        const Real dy = Real(1) / ((Real(1) + yx * yx) * x.n());
        const Real dx = dy * (-yx);
        return binary(x, y, atan2(x.n(), y.n()), dx, dy);
    }
    template<typename Real>
    UReal<Real> sinh(const UReal<Real> &x) {
        using std::sinh;
        using std::cosh;
        return unary(x, sinh(x.n()), cosh(x.n()));
    }
    template<typename Real>
    UReal<Real> cosh(const UReal<Real> &x) {
        using std::cosh;
        using std::sinh;
        return unary(x, cosh(x.n()), sinh(x.n()));
    }
    template<typename Real>
    UReal<Real> tanh(const UReal<Real> &x) {
        using std::tanh;
        const Real t = tanh(x.n());
        return unary(x, t, Real(1) - t * t);
    }
    template<typename Real>
    UReal<Real> asinh(const UReal<Real> &x) {
        using std::asinh;
        using std::sqrt;
        return unary(x, asinh(x.n()), Real(1) / sqrt(x.n() * x.n() + 1));
    }
    template<typename Real>
    UReal<Real> acosh(const UReal<Real> &x) {
        using std::acosh;
        using std::sqrt;
        return unary(x, acosh(x.n()), Real(1) / sqrt(x.n() * x.n() - 1));
    }
    template<typename Real>
    UReal<Real> atanh(const UReal<Real> &x) {
        using std::atanh;
        return unary(x, atanh(x.n()), Real(1) / (1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> erf(const UReal<Real> &x) {
        using std::erf;
        using std::sqrt;
        using std::exp;
        using std::atan2;
        static const Real erf_coeff = Real(2) / sqrt(atan2(Real(0), Real(-1)));
        return unary(x, erf(x.n()), erf_coeff * exp(-x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> erfc(const UReal<Real> &x) {
        using std::erfc;
        using std::sqrt;
        using std::exp;
        using std::atan2;
        static const Real erf_coeff = Real(2) / sqrt(atan2(Real(0), Real(-1)));
        return unary(x, erfc(x.n()), -erf_coeff * exp(-x.n() * x.n()));
    }
    /*!
    \brief Return `isfinite(x.n()) && isfinite(x.s())`.
    */
    template<typename Real>
    bool isfinite(const UReal<Real> &x) {
        using std::isfinite;
        return isfinite(x.n()) and isfinite(x.s());
    }
    /*!
    \brief Return `isnormal(x.n()) && isnormal(x.s())`.
    */
    template<typename Real>
    bool isnormal(const UReal<Real> &x) {
        using std::isnormal;
        return isnormal(x.n()) and isnormal(x.s());
    }




    template<typename Real, Prop prop>
    UReal2<Real, prop> abs(const UReal2<Real, prop> &x) {
        using std::abs;
        const Real &n = x.first_order_n();
        return unary(x, abs(n), n >= 0 ? 1 : -1, 0);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> fmod(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::fmod;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        return binary(x, y, fmod(xn, yn), 1, -trunc(xn / yn), 0, 0, 0);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> remainder(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::remainder;
        using std::round;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        return binary(x, y, remainder(xn, yn), 1, -round(xn / yn), 0, 0, 0);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> fmax(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::fmax;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        const Real max = fmax(xn, yn);
        const bool c = max == xn;
        return binary(x, y, max, c ? 1 : 0, c ? 0 : 1, 0, 0, 0);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> fmin(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::fmin;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        const Real min = fmin(xn, yn);
        const bool c = min == xn;
        return binary(x, y, min, c ? 1 : 0, c ? 0 : 1, 0, 0, 0);
    }
    // fdim?
    template<typename Real, Prop prop>
    UReal2<Real, prop> exp(const UReal2<Real, prop> &x) {
        using std::exp;
        const Real &xn = x.first_order_n();
        return unary(x, exp(xn), exp(xn), exp(xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> exp2(const UReal2<Real, prop> &x) {
        using std::exp2;
        using std::log;
        const Real &xn = x.first_order_n();
        static const Real coeff = log(Real(2));
        return unary(x, exp2(xn), coeff * exp2(xn), coeff * coeff * exp2(xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> expm1(const UReal2<Real, prop> &x) {
        using std::expm1;
        using std::exp;
        const Real &xn = x.first_order_n();
        return unary(x, expm1(xn), exp(xn), exp(xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> log(const UReal2<Real, prop> &x) {
        using std::log;
        const Real &xn = x.first_order_n();
        return unary(x, log(xn), 1 / xn, -1 / (xn * xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> log10(const UReal2<Real, prop> &x) {
        using std::log10;
        using std::log;
        const Real &xn = x.first_order_n();
        static const Real coeff = 1 / log(Real(10));
        return unary(x, log10(xn), coeff / xn, -coeff / (xn * xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> log2(const UReal2<Real, prop> &x) {
        using std::log2;
        using std::log;
        const Real &xn = x.first_order_n();
        static const Real coeff = 1 / log(Real(2));
        return unary(x, log2(xn), coeff / xn, -coeff / (xn * xn));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> log1p(const UReal2<Real, prop> &x) {
        using std::log1p;
        const Real &xn = x.first_order_n();
        const Real ixnp1 = 1 / (1 + xn);
        return unary(x, log1p(xn), ixnp1, -ixnp1 * ixnp1);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> pow(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::pow;
        using std::log;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        const Real x2y = pow(xn, yn);
        const Real logx = log(xn);
        const Real ix = 1 / xn;
        const Real x2yix = x2y * ix;
        const Real x2yixy = x2yix * yn;
        const Real x2ylogx = x2y * logx;
        return binary(x, y, x2y,
                      x2yixy,
                      x2ylogx,
                      x2yixy * ix * (yn - 1),
                      x2ylogx * logx,
                      x2yix + x2yixy * logx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> sqrt(const UReal2<Real, prop> &x) {
        using std::sqrt;
        const Real &xn = x.first_order_n();
        const Real sqrtx = sqrt(xn);
        return unary(x, sqrtx, 1 / (2 * sqrtx), -1 / (4 * xn * sqrtx));
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> cbrt(const UReal2<Real, prop> &x) {
        using std::cbrt;
        using std::pow;
        const Real &xn = x.first_order_n();
        const Real cbrtx = cbrt(xn);
        const Real icbrt2 = 1 / (cbrtx * cbrtx);
        return unary(x, cbrtx, icbrt2 / 3, -2 / (9 * xn) * icbrt2);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> hypot(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::hypot;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        const Real h = hypot(xn, yn);
        const Real ih = 1 / h;
        const Real xnih = xn * ih;
        const Real ynih = yn * ih;
        return binary(x, y, h,
                      xnih, ynih,
                      ih * (1 - xnih * xnih),
                      ih * (1 - ynih * ynih),
                      -xnih * ynih * ih);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> sin(const UReal2<Real, prop> &x) {
        using std::sin;
        using std::cos;
        const Real &xn = x.first_order_n();
        const Real sinx = sin(xn);
        return unary(x, sinx, cos(xn), -sinx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> cos(const UReal2<Real, prop> &x) {
        using std::cos;
        using std::sin;
        const Real &xn = x.first_order_n();
        const Real cosx = cos(xn);
        return unary(x, cosx, -sin(xn), -cosx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> tan(const UReal2<Real, prop> &x) {
        using std::tan;
        const Real &xn = x.first_order_n();
        const Real t = tan(xn);
        const Real tt1 = 1 + t * t;
        return unary(x, t, tt1, 2 * t * tt1);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> asin(const UReal2<Real, prop> &x) {
        using std::asin;
        using std::sqrt;
        const Real &xn = x.first_order_n();
        const Real i1xx = 1 / (1 - xn * xn);
        const Real isqrt1xx = sqrt(i1xx);
        return unary(x, asin(xn), isqrt1xx, xn * isqrt1xx * i1xx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> acos(const UReal2<Real, prop> &x) {
        using std::acos;
        using std::sqrt;
        const Real &xn = x.first_order_n();
        const Real i1xx = 1 / (1 - xn * xn);
        const Real isqrt1xx = sqrt(i1xx);
        return unary(x, acos(xn), -isqrt1xx, -xn * isqrt1xx * i1xx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> atan(const UReal2<Real, prop> &x) {
        using std::atan;
        const Real &xn = x.first_order_n();
        const Real i1xx = 1 / (1 + xn * xn);
        return unary(x, atan(xn), i1xx, -2 * xn * i1xx * i1xx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> atan2(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        using std::atan2;
        const Real &xn = x.first_order_n();
        const Real &yn = y.first_order_n();
        const Real x2 = xn * xn;
        const Real y2 = yn * yn;
        const Real xxyy = x2 + y2;
        const Real ixxyy = 1 / xxyy;
        const Real xixxyy = xn * ixxyy;
        const Real yixxyy = yn * ixxyy;
        const Real xy2ixxyy2 = 2 * xixxyy * yixxyy;
        return binary(x, y, atan2(xn, yn),
                      yixxyy,
                      -xixxyy,
                      -xy2ixxyy2,
                      xy2ixxyy2,
                      (x2 - y2) * ixxyy * ixxyy);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> sinh(const UReal2<Real, prop> &x) {
        using std::sinh;
        using std::cosh;
        const Real &xn = x.first_order_n();
        const Real sinhx = sinh(xn);
        return unary(x, sinhx, cosh(xn), sinhx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> cosh(const UReal2<Real, prop> &x) {
        using std::cosh;
        using std::sinh;
        const Real &xn = x.first_order_n();
        const Real coshx = cosh(xn);
        return unary(x, coshx, sinh(xn), coshx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> tanh(const UReal2<Real, prop> &x) {
        using std::tanh;
        const Real &xn = x.first_order_n();
        const Real t = tanh(xn);
        const Real tt1 = 1 - t * t;
        return unary(x, t, tt1, -2 * t * tt1);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> asinh(const UReal2<Real, prop> &x) {
        using std::asinh;
        using std::sqrt;
        const Real &xn = x.first_order_n();
        const Real i1xx = 1 / (1 + xn * xn);
        const Real sqrti1xx = sqrt(i1xx);
        return unary(x, asinh(xn), sqrti1xx, -xn * i1xx * sqrti1xx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> acosh(const UReal2<Real, prop> &x) {
        using std::acosh;
        using std::sqrt;
        const Real &xn = x.first_order_n();
        const Real ixx1 = 1 / (xn * xn - 1);
        const Real sqrtixx1 = sqrt(ixx1);
        return unary(x, acosh(xn), sqrtixx1, -xn * ixx1 * sqrtixx1);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> atanh(const UReal2<Real, prop> &x) {
        using std::atanh;
        const Real &xn = x.first_order_n();
        const Real i1xx = 1 / (1 - xn * xn);
        return unary(x, atanh(xn), i1xx, 2 * xn * i1xx * i1xx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> erf(const UReal2<Real, prop> &x) {
        using std::erf;
        using std::sqrt;
        using std::exp;
        using std::atan2;
        static const Real erf_coeff = 2 / sqrt(atan2(Real(0), Real(-1)));
        const Real &xn = x.first_order_n();
        const Real cexx = erf_coeff * exp(-xn * xn);
        return unary(x, erf(xn), cexx, -2 * xn * cexx);
    }
    template<typename Real, Prop prop>
    UReal2<Real, prop> erfc(const UReal2<Real, prop> &x) {
        using std::erfc;
        using std::sqrt;
        using std::exp;
        using std::atan2;
        static const Real erf_coeff = 2 / sqrt(atan2(Real(0), Real(-1)));
        const Real &xn = x.first_order_n();
        const Real cexx = erf_coeff * exp(-xn * xn);
        return unary(x, erfc(xn), -cexx, 2 * xn * cexx);
    }
    template<typename Real, Prop prop>
    bool isfinite(const UReal2<Real, prop> &x) {
        using std::isfinite;
        return isfinite(x.n()) and isfinite(x.s());
    }
    template<typename Real, Prop prop>
    bool isnormal(const UReal2<Real, prop> &x) {
        using std::isnormal;
        return isnormal(x.n()) and isnormal(x.s());
    }
}

#endif /* end of include guard: UNCERTAINTIES_MATH_HPP_0A89E052 */
