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

To propagate the uncertainty, the derivative of the function is needed.
The functions in this file pack together a function and its derivative(s) in
the appropriate way, or take only the function and compute the derivative
themselves.

Example re-implementing `sin` with explicit derivative:

~~~cpp
#include <cmath>
#include <uncertainties/ureal.hpp>
#include <uncertainties/functions.hpp>
...
namespace unc = uncertainties;
unc::UBinary<double> usin;
usin = unc::ubinary<double>([](double x) { return sin(x); },
                            [](double x) { return cos(x); });
unc::udouble x = {1, 0.1};
unc::udouble y = usin(x);
~~~

Example re-implementing `sin` with automatical derivative:

~~~cpp
unc::UBinary<double> usin_num;
usin_num = unc::ubinary<double>([](double x) { return sin(x); });
y = usin_num(x);
~~~

Note: automatical derivatives implemented here are very simple and significantly
less precise. You may encounter computational problems if you use automatical
derivatives, in particular when using `float`.

*/

#include <functional>
#include <limits>
#include <cmath>

#include "core.hpp"

namespace uncertainties {
    /*!
    \brief Type of a function object taking one `Real` and returning `Real`.
    */
    template<typename Real>
    using Unary = std::function<Real(const Real &)>;
    
    /*!
    \brief Type of a function object taking two `Real`s and returning `Real`.
    */
    template<typename Real>
    using Binary = std::function<Real(const Real &, const Real &)>;

    /*!
    \brief Type of a function object taking one `UReal` and returning `UReal`.
    */
    template<typename Real>
    using UUnary = std::function<UReal<Real>(const UReal<Real> &)>;
    
    /*!
    \brief Type of a function object taking two `UReal`s and returning `UReal`.
    */
    template<typename Real>
    using UBinary = std::function<UReal<Real>(const UReal<Real> &, const UReal<Real> &)>;
    
    /*!
    \brief Construct a function on `UReal`s of one argument.
    
    `f` is the function to compute and `df` its derivative.
    */
    template<typename Real>
    UUnary<Real> uunary(const Unary<Real> &f, const Unary<Real> &df) {
        return [f, df](const UReal<Real> &x) {
            return unary(x, f(x.n()), df(x.n()));
        };
    }
    
    /*!
    \brief Construct a function on `UReal`s of two arguments.
    
    `f` is the function to compute, `dfdx` its derivative respect to the first
    argument and `dfdy` respect to the second.
    */
    template<typename Real>
    UBinary<Real>
    ubinary(const Binary<Real> &f,
            const Binary<Real> &dfdx, const Binary<Real> &dfdy) {
        return [f, dfdx, dfdy](const UReal<Real> &x, const UReal<Real> &y) {
            const Real &xn = x.n();
            const Real &yn = y.n();
            return binary(x, y, f(xn, yn), dfdx(xn, yn), dfdy(xn, yn));
        };
    }
    
    /*!
    \brief Default step for forward difference derivatives.
    */
    template<typename Real>
    Real default_step(const int order=1) {
        using std::sqrt;
        using std::numeric_limits;
        return order > 0 ? sqrt(default_step<Real>(order - 1)) : numeric_limits<Real>::epsilon();
    };
    
    /*!
    \brief Construct a function on `UReal`s of one argument.
    
    `f` is the function to compute. The derivative is computed with a forward
    difference with step `step`:
    
    \f{align*}{
    f'(x) &= \frac {f(x + \mathrm{step}) - f(x)} {\mathrm{step}}
    \f}
    */
    template<typename Real>
    UUnary<Real>
    uunary(const Unary<Real> &f,
           const Real &step=default_step<Real>()) {
        return [f, step](const UReal<Real> &x) {
            const Real &mu = x.n();
            const Real fmu = f(mu);
            using std::abs;
            const Real dx = (f(mu + step) - fmu) / step;
            return unary(x, fmu, dx);
        };
    }
    
    /*!
    \brief Construct a function on `UReal`s of two arguments.
    
    `f` is the function to compute. The derivatives are computed with a forward
    difference with step `step`:
    
    \f{align*}{
    \frac {\partial f} {\partial x} (x, y)
    &= \frac {f(x + \mathrm{step}, y) - f(x, y)} {\mathrm{step}} \\
    \frac {\partial f} {\partial y} (x, y)
    &= \frac {f(x, y + \mathrm{step}) - f(x, y)} {\mathrm{step}}
    \f}
    */
    template<typename Real>
    UBinary<Real>
    ubinary(const Binary<Real> &f,
            const Real &step=default_step<Real>()) {
        return [f, step](const UReal<Real> &x, const UReal<Real> &y) {
            const Real &xn = x.n();
            const Real &yn = y.n();
            const Real fn = f(xn, yn);
            using std::abs;
            const Real dfdx = (f(xn + step, yn) - fn) / step;
            const Real dfdy = (f(xn, yn + step) - fn) / step;
            return binary(x, y, fn, dfdx, dfdy);
        };
    }



    /*!
    \brief Type of a function object taking one `UReal2` and returning `UReal2`.
    */
    template<typename Real, Prop prop>
    using UUnary2 = std::function<UReal2<Real, prop>(const UReal2<Real, prop> &)>;
    
    /*!
    \brief Type of a function object taking two `UReal2`s and returning `UReal2`.
    */
    template<typename Real, Prop prop>
    using UBinary2 = std::function<UReal2<Real, prop>(
        const UReal2<Real, prop> &, const UReal2<Real, prop> &
    )>;
    
    /*!
    \brief Construct a function on `UReal2`s of one argument.
    */
    template<typename Real, Prop prop>
    UUnary2<Real, prop> uunary2(const Unary<Real> &f,
                                const Unary<Real> &dfdx,
                                const Unary<Real> &ddfdxdx) {
        return [=](const UReal2<Real, prop> &x) {
            const Real &xn = x.first_order_n();
            return unary(x, f(xn), dfdx(xn), ddfdxdx(xn));
        };
    }
    
    /*!
    \brief Construct a function on `UReal2`s of two arguments.
    */
    template<typename Real, Prop prop>
    UBinary2<Real, prop> ubinary2(const Binary<Real> &f,
                                  const Binary<Real> &dfdx,
                                  const Binary<Real> &dfdy,
                                  const Binary<Real> &ddfdxdx,
                                  const Binary<Real> &ddfdydy,
                                  const Binary<Real> &ddfdxdy) {
        return [=](const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            const Real &xn = x.first_order_n();
            const Real &yn = y.first_order_n();
            return binary(x, y, f(xn, yn),
                          dfdx(xn, yn), dfdy(xn, yn),
                          ddfdxdx(xn, yn), ddfdydy(xn, yn), ddfdxdy(xn, yn));
        };
    }
        
    /*!
    \brief Construct a function on `UReal2`s of one argument.
    
    `f` is the function to compute. The derivative is computed with a forward
    difference with step `step`:
    
    \f{align*}{
    f'(x) &= \frac {f(x + \mathrm{step}) - f(x)} {\mathrm{step}}
    \f}
    */
    template<typename Real, Prop prop>
    UUnary2<Real, prop>
    uunary2(const Unary<Real> &f,
            const Unary<Real> &dfdx,
            const Real &step=default_step<Real>()) {
        return [=](const UReal2<Real, prop> &x) {
            const Real &xn = x.first_order_n();
            const Real dfdxn = dfdx(xn);
            const Real ddfdxdx = (dfdx(xn + step) - dfdxn) / step;
            return unary(x, f(xn), dfdxn, ddfdxdx);
        };
    }
    
    /*!
    \brief Construct a function on `UReal2`s of one argument.
    
    `f` is the function to compute. The derivative is computed with a forward
    difference with step `step`:
    
    \f{align*}{
    f'(x) &= \frac {f(x + \mathrm{step}) - f(x)} {\mathrm{step}}
    \f}
    */
    template<typename Real, Prop prop>
    UUnary2<Real, prop>
    uunary2(const Unary<Real> &f,
            const Real &step=default_step<Real>(2)) {
        return [=](const UReal2<Real, prop> &x) {
            const Real &xn = x.first_order_n();
            const Real fc = f(xn);
            const Real fr = f(xn + step);
            const Real fl = f(xn - step);
            const Real dfdx = (fr - fl) / (2 * step);
            const Real ddfdxdx = (fr + fl - 2 * fc) / (step * step);
            return unary(x, fc, dfdx, ddfdxdx);
        };
    }
    
    /*!
    \brief Construct a function on `UReal2`s of two arguments.
    
    `f` is the function to compute. The derivatives are computed with a forward
    difference with step `step`:
    
    \f{align*}{
    \frac {\partial f} {\partial x} (x, y)
    &= \frac {f(x + \mathrm{step}, y) - f(x, y)} {\mathrm{step}} \\
    \frac {\partial f} {\partial y} (x, y)
    &= \frac {f(x, y + \mathrm{step}) - f(x, y)} {\mathrm{step}}
    \f}
    */
    template<typename Real, Prop prop>
    UBinary2<Real, prop>
    ubinary2(const Binary<Real> &f,
             const Binary<Real> &dfdx,
             const Binary<Real> &dfdy,
             const Real &step=default_step<Real>()) {
        return [=](const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            const Real &xn = x.first_order_n();
            const Real &yn = y.first_order_n();
            const Real dfdxn = dfdx(xn, yn);
            const Real dfdyn = dfdy(xn, yn);
            const Real ddfdxdx = (dfdx(xn + step, yn) - dfdxn) / step;
            const Real ddfdydy = (dfdy(xn, yn + step) - dfdyn) / step;
            const Real ddfdxdy = (dfdx(xn, yn + step) - dfdxn) / step;
            return binary(x, y, f(xn, yn),
                          dfdxn, dfdyn,
                          ddfdxdx, ddfdydy, ddfdxdy);
        };
    }

    /*!
    \brief Construct a function on `UReal2`s of two arguments.
    
    `f` is the function to compute. The derivatives are computed with a forward
    difference with step `step`:
    
    \f{align*}{
    \frac {\partial f} {\partial x} (x, y)
    &= \frac {f(x + \mathrm{step}, y) - f(x, y)} {\mathrm{step}} \\
    \frac {\partial f} {\partial y} (x, y)
    &= \frac {f(x, y + \mathrm{step}) - f(x, y)} {\mathrm{step}}
    \f}
    */
    template<typename Real, Prop prop>
    UBinary2<Real, prop>
    ubinary2(const Binary<Real> &f,
             const Real &step=default_step<Real>(2)) {
        return [=](const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            const Real &xc = x.first_order_n();
            const Real &yc = y.first_order_n();
            const Real fcc = f(xc, yc);
            const Real xr = xc + step;
            const Real xl = xc - step;
            const Real yr = yc + step;
            const Real yl = yc - step;
            const Real frc = f(xr, yc);
            const Real flc = f(xl, yc);
            const Real fcr = f(xc, yr);
            const Real fcl = f(xc, yl);
            const Real frr = f(xr, yr);
            const Real fll = f(xl, yl);
            const Real frclc = frc + flc;
            const Real fcrcl = fcr + fcl;
            const Real fcc2 = 2 * fcc;
            const Real step2 = 2 * step;
            const Real stepsq = step * step;
            const Real dfdx = (frc - flc) / step2;
            const Real dfdy = (fcr - fcl) / step2;
            const Real ddfdxdx = (frclc - fcc2) / stepsq;
            const Real ddfdydy = (fcrcl - fcc2) / stepsq;
            const Real ddfdxdy = (frr + fll - frclc - fcrcl + fcc2) / (2 * stepsq);
            return binary(x, y, fcc,
                          dfdx, dfdy,
                          ddfdxdx, ddfdydy, ddfdxdy);
        };
    }
}

#endif /* end of include guard: UNCERTAINTIES_FUNCTIONS_HPP_7FF782C8 */
