// functions.hpp
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

#ifndef UNCERTAINTIES_FUNCTIONS_HPP_7FF782C8
#define UNCERTAINTIES_FUNCTIONS_HPP_7FF782C8

/*! \file
\brief Factories to create functions taking `UReal`s as input.
*/

#include <functional>
#include <limits>
#include <cmath>

#include "core.hpp"

namespace uncertainties {
    template<typename Real>
    std::function<UReal<Real>(const UReal<Real> &)>
    uunary(const std::function<const Real &(const Real &)> &f,
           const std::function<const Real &(const Real &)> &df) {
        return [f, df](const UReal<Real> &x) {
            return unary(x, f(x.n()), df(x.n()));
        };
    }
    
    template<typename Real>
    std::function<UReal<Real>(const UReal<Real> &, const UReal<Real> &)>
    ubinary(const std::function<Real(const Real &, const Real &)> &f,
            const std::function<Real(const Real &, const Real &)> &dfdx,
            const std::function<Real(const Real &, const Real &)> &dfdy) {
        return [f, dfdx, dfdy](const UReal<Real> &x, const UReal<Real> &y) {
            const Real &xn = x.n();
            const Real &yn = y.n();
            return binary(x, y, f(xn, yn), dfdx(xn, yn), dfdy(xn, yn));
        };
    }
    
    namespace internal {
        template<typename Real>
        constexpr Real default_step() {
            using std::sqrt;
            return sqrt(std::numeric_limits<Real>::epsilon());
        };
    }
    
    template<typename Real>
    std::function<UReal<Real>(const UReal<Real> &)>
    uunary(const std::function<Real(const Real &)> &f,
           const Real &astep=internal::default_step<Real>(),
           const Real &rstep=internal::default_step<Real>()) {
        return [f, rstep, astep](const UReal<Real> &x) {
            const Real &mu = x.n();
            const Real fmu = f(mu);
            using std::abs;
            const Real step = abs(fmu) * rstep + astep;
            const Real dx = (f(mu + step) - fmu) / step;
            return unary(x, fmu, dx);
        };
    }
    
    template<typename Real>
    std::function<UReal<Real>(const UReal<Real> &, const UReal<Real> &)>
    ubinary(const std::function<Real(const Real &, const Real &)> &f,
            const Real &astep=internal::default_step<Real>(),
            const Real &rstep=internal::default_step<Real>()) {
        return [f, rstep, astep](const UReal<Real> &x, const UReal<Real> &y) {
            const Real &xn = x.n();
            const Real &yn = y.n();
            const Real fn = f(xn, yn);
            using std::abs;
            const Real step = abs(fn) * rstep + astep;
            const Real dfdx = (f(xn + step, yn) - fn) / step;
            const Real dfdy = (f(xn, yn + step) - fn) / step;
            return binary(x, y, fn, dfdx, dfdy);
        };
    }
}

#endif /* end of include guard: UNCERTAINTIES_FUNCTIONS_HPP_7FF782C8 */
