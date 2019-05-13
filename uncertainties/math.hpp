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
\brief Defines functions of `<cmath>` on `UReal`s.
*/

#include <cmath>

#include "core.hpp"

namespace uncertainties {
    template<typename Real>
    UReal<Real> abs(const UReal<Real> &x) {
        return unary(x, std::abs(x.n()), x.n() >= 0 ? 1 : -1);
    }
    template<typename Real>
    UReal<Real> fmod(const UReal<Real> &x, const UReal<Real> &y) {
        return binary(x, y, std::fmod(x.n(), y.n()), 1, -std::trunc(x.n() / y.n()));
    }
    template<typename Real>
    UReal<Real> remainder(const UReal<Real> &x, const UReal<Real> &y) {
        return binary(x, y, std::remainder(x.n(), y.n()), 1, -std::round(x.n() / y.n()));
    }
    template<typename Real>
    UReal<Real> fmax(const UReal<Real> &x, const UReal<Real> &y) {
        const Real max = std::fmax(x.n(), y.n());
        const bool c = max == x.n();
        return binary(x, y, max, c ? 1 : 0, c ? 0 : 1);
    }
    template<typename Real>
    UReal<Real> fmin(const UReal<Real> &x, const UReal<Real> &y) {
        const Real min = std::fmin(x.n(), y.n());
        const bool c = min == x.n();
        return binary(x, y, min, c ? 1 : 0, c ? 0 : 1);
    }
    // fdim?
    template<typename Real>
    UReal<Real> exp(const UReal<Real> &x) {
        return unary(x, std::exp(x.n()), std::exp(x.n()));
    }
    template<typename Real>
    UReal<Real> exp2(const UReal<Real> &x) {
        return unary(x, std::exp2(x.n()), std::log(Real(2)) * std::exp2(x.n()));
    }
    template<typename Real>
    UReal<Real> expm1(const UReal<Real> &x) {
        return unary(x, std::expm1(x.n()), std::exp(x.n()));
    }
    template<typename Real>
    UReal<Real> log(const UReal<Real> &x) {
        return unary(x, std::log(x.n()), Real(1) / x.n());
    }
    template<typename Real>
    UReal<Real> log10(const UReal<Real> &x) {
        return unary(x, std::log10(x.n()), Real(1) / (x.n() * std::log(Real(10))));
    }
    template<typename Real>
    UReal<Real> log2(const UReal<Real> &x) {
        return unary(x, std::log2(x.n()), Real(1) / (x.n() * std::log(Real(2))));
    }
    template<typename Real>
    UReal<Real> log1p(const UReal<Real> &x) {
        return unary(x, std::log1p(x.n()), Real(1) / (Real(1) + x.n()));
    }
    template<typename Real>
    UReal<Real> pow(const UReal<Real> &x, const UReal<Real> &y) {
        const Real p = std::pow(x.n(), y.n());
        return binary(x, y, p, p * y.n() / x.n(), p * std::log(x.n()));
    }
    template<typename Real>
    UReal<Real> sqrt(const UReal<Real> &x) {
        return unary(x, std::sqrt(x.n()), Real(1) / (2 * std::sqrt(x.n())));
    }
    template<typename Real>
    UReal<Real> cbrt(const UReal<Real> &x) {
        return unary(x, std::cbrt(x.n()), std::pow(x.n(), -Real(2) / Real(3)) / Real(3));
    }
    template<typename Real>
    UReal<Real> hypot(const UReal<Real> &x, const UReal<Real> &y) {
        const Real h = std::hypot(x.n(), y.n());
        return binary(x, y, h, x.n() / h, y.n() / h);
    }
    template<typename Real>
    UReal<Real> sin(const UReal<Real> &x) {
        return unary(x, std::sin(x.n()), std::cos(x.n()));
    }
    template<typename Real>
    UReal<Real> cos(const UReal<Real> &x) {
        return unary(x, std::cos(x.n()), -std::sin(x.n()));
    }
    template<typename Real>
    UReal<Real> tan(const UReal<Real> &x) {
        const Real t = std::tan(x.n());
        return unary(x, t, Real(1) + t * t);
    }
    template<typename Real>
    UReal<Real> asin(const UReal<Real> &x) {
        return unary(x, std::asin(x.n()), Real(1) / std::sqrt(1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> acos(const UReal<Real> &x) {
        return unary(x, std::acos(x.n()), -Real(1) / std::sqrt(1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> atan(const UReal<Real> &x) {
        return unary(x, std::atan(x.n()), Real(1) / (1 + x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> atan2(const UReal<Real> &x, const UReal<Real> &y) {
        const Real yx = y.n() / x.n();
        const Real dy = Real(1) / ((Real(1) + yx * yx) * x.n());
        const Real dx = dy * (-yx);
        return binary(x, y, std::atan2(x.n(), y.n()), dx, dy);
    }
    template<typename Real>
    UReal<Real> sinh(const UReal<Real> &x) {
        return unary(x, std::sinh(x.n()), std::cosh(x.n()));
    }
    template<typename Real>
    UReal<Real> cosh(const UReal<Real> &x) {
        return unary(x, std::cosh(x.n()), std::sinh(x.n()));
    }
    template<typename Real>
    UReal<Real> tanh(const UReal<Real> &x) {
        const Real t = std::tanh(x.n());
        return unary(x, t, Real(1) - t * t);
    }
    template<typename Real>
    UReal<Real> asinh(const UReal<Real> &x) {
        return unary(x, std::asinh(x.n()), Real(1) / std::sqrt(x.n() * x.n() + 1));
    }
    template<typename Real>
    UReal<Real> acosh(const UReal<Real> &x) {
        return unary(x, std::acosh(x.n()), Real(1) / std::sqrt(x.n() * x.n() - 1));
    }
    template<typename Real>
    UReal<Real> atanh(const UReal<Real> &x) {
        return unary(x, std::atanh(x.n()), Real(1) / (1 - x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> erf(const UReal<Real> &x) {
        static const Real erf_coeff = Real(2) / std::sqrt(Real(3.141592653589793238462643383279502884L));
        return unary(x, std::erf(x.n()), erf_coeff * std::exp(-x.n() * x.n()));
    }
    template<typename Real>
    UReal<Real> erfc(const UReal<Real> &x) {
        static const Real erf_coeff = Real(2) / std::sqrt(Real(3.141592653589793238462643383279502884L));
        return unary(x, std::erfc(x.n()), -erf_coeff * std::exp(-x.n() * x.n()));
    }
    template<typename Real>
    bool isfinite(const UReal<Real> &x) {
        return std::isfinite(x.n()) and std::isfinite(x.s());
    }
    template<typename Real>
    bool isnormal(const UReal<Real> &x) {
        return std::isnormal(x.n()) and std::isnormal(x.s());
    }
}

#endif /* end of include guard: UNCERTAINTIES_MATH_HPP_0A89E052 */
