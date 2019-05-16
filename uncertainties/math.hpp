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
}

#endif /* end of include guard: UNCERTAINTIES_MATH_HPP_0A89E052 */
