// ureal2.hpp
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

#ifndef UNCERTAINTIES_UREAL2_HPP_4F22829C
#define UNCERTAINTIES_UREAL2_HPP_4F22829C

/*! \file
\brief Defines class template `UReal2`.
*/

#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <utility>
#include <memory>
#include <cassert>
#include <limits>
#include <sstream>

#include "core.hpp"
#include "internal/hessgrad.hpp"
#include "internal/moments.hpp"
#include "internal/checkmoments.hpp"

namespace uncertainties {
    namespace internal {
        inline int propsign(const Prop prop) noexcept {
            return prop == Prop::est ? -1 : 1;
        }
        
        template<typename Real>
        void update_mom(std::array<Real, 4> &mom,
                        const int i,
                        const HessGrad<Real> &hg) noexcept {
            assert(i >= 0 and i <= 3);
            const Real &m1 = mom[0];
            const Real &m2 = mom[1];
            const Real &m3 = mom[2];
            const Real m = internal::compute_mom(hg, i + 1);
            switch (i) {
            case 0:
                mom[i] = m;
                break;
            case 1:
                mom[i] = m - m1 * m1;
                break;
            case 2:
                mom[i] = m - m1 * (3*m2 + m1*m1);
                break;
            case 3:
                mom[i] = m - m1 * (4*m3 + m1 * (6*m2 + m1*m1));
            }
        }
    }
    
    /*!
    \brief Class for second order error propagation.
    
    Theory
    ======
    
    It may seem pedantic to start the documentation of a class with
    mathematical theory, but second-order uncertainty propagation has
    additional caveats compared to first-order propagation that require user
    awareness right from the beginning. In particular, you may have noticed that
    there is a template parameter `prop` with no default value. What should
    you do with that? You can not leave it unspecified if you want to use the
    class. So, read on.
    
    Recap on moments
    ----------------
    
    Let \f$ x \f$ be a real random variable, i.e. a variable with an associated
    probability density function \f$ p(x) \ge 0 \f$. The _expected value_ of
    a function \f$ f(x) \f$ with respect to the variable \f$ x \f$ is defined as
    
    \f[
    E[f(x)] = \int_{-\infty}^{\infty} \mathrm{d}x\, p(x)f(x).
    \f]
    
    The intuitive meaning of the expected value is an average: it sums up
    all the possible values of \f$ f(x) \f$ weighting them with the
    probability \f$ p(x)\mathrm{d}x \f$.
    
    For the particular choice \f$ f(x) = x^n \f$, one obtains the _moments_ of
    the distribution \f$ p(x) \f$:
    
    \f[
    m_n = E[x^n].
    \f]
    
    The moment \f$ m_1 = E[x] \f$ is the _mean_ of the distribution. If one
    translates \f$ x \f$ to have zero mean, the new moments are called
    _central moments_:
    
    \f[
    \mu_n = E[(x - E[x])^n].
    \f]
    
    The _variance_ is \f$ \mu_2 \f$, and the _standard deviation_ is \f$ \sigma
    = \sqrt{\mu_2} \f$. By rescaling the \f$ n \f$th central moment with the
    \f$ n \f$th power of \f$ \sigma \f$, we obtain the _standardized central
    moments_:
    
    \f[
    k_n = \frac {\mu_n} {\sigma^n} \quad (\sigma = \sqrt{\mu_2}).
    \f]
    
    Observation: by definition \f$ m_0 = \mu_0 = k_0 = 1 \f$ (because \f$ p(x)
    \f$ is normalized), \f$ \mu_1 = k_1 = 0 \f$, and \f$ k_2 = 1 \f$.
    
    The first moments have an intuitive interpretation: the standard deviation
    \f$ \sigma \f$ gives a measure of the width of the distribution, and is
    commonly used as the "error", the _skewness_ \f$ k_3 \f$ indicates if the
    distribution is asymmetric (\f$ k_3 > 0 \f$ mean distribution with tail on
    the right, \f$ k_3 < 0 \f$ on the left), the _kurtosis_ \f$ k_4 \f$
    indicates that the distribution has fat tails, if large, and suggests that
    the distribution may be bimodal, if small. To have an idea of what "small"
    and "large" mean for the kurtosis, consider that it can be shown that \f$
    k_4 \ge 1 + k_3^2 \f$ and that for a normal (gaussian) distribution, \f$
    k_4 = 3 \f$.
    
    Moment propagation
    ------------------
    
    Let \f$ x \f$ be a random variable with central moments \f$ \mu_n \f$. We
    can always translate \f$ x \f$ to have zero mean, such that
    \f$ \mu_n = E[x^n] \f$ (just to simplify calculations).
    
    Let \f$ y \f$ be a random variable defined by \f$ y = f(x) \f$. We expand
    \f$ f \f$ in Taylor series up to second order around the mean of \f$ x \f$
    (which is zero):
    
    \f[
    f(x) = f(0) + f'(0) x + \frac12 f''(0) x^2 + O(x^3).
    \f]
    
    Again, we can traslate \f$ y \f$ to have \f$ f(0) = 0 \f$, and translate
    back later. We now compute the moments of \f$ y \f$ using the expansion.
    First the mean:
    
    \f{align*}{
    E[y] &= E[f(x)] \approx \\
    &\approx E \left[ f'(0) x + \frac12 f''(0) x^2 \right] = \\
    &= \frac12 f''(0) E[x^2].
    \f}
    
    (We have used the properties that \f$ E[A + B] = E[A] + E[B] \f$ and
    \f$ E[\alpha A(x)] = \alpha E[A(x)] \f$.)
    
    Computing the variance \f$ E[(y-E[y])^2] \f$ is cumbersome, so we compute
    \f$ E[y^2] \f$ and observe that \f$ E[(y-E[y])^2] = E[y^2] - E[y]^2 \f$.
    We also drop the zero in \f$ f'(0) \f$ and \f$ f''(0) \f$ for brevity:
    
    \f{align*}{
    E[y^2]
    &\approx E\left[ \left( f'x + \frac12 f''x^2 \right)^2 \right] = \\
    &= E\left[ f'f'x^2 + f'f''x^3 + \frac14 f''f''x^4 \right] = \\
    &= f'f'E[x^2] + f'f''E[x^3] + \frac14 f''f''E[x^4].
    \f}
    
    We can go on to arbitrary moments and also generalize to the case of many
    variables (\f$ y = f(x_1, x_2, \ldots) \f$), but the reader is probably
    already bored enough. These computations will be all carried on by the code
    under the hood. Just remember the following two things.
    
    First thing: let's write the formula for the mean of \f$ y \f$ by
    translating back everything:
    
    \f{align}{
    E[y] &\approx f(E[x]) + \frac12 f''(E[x]) E[(x-E[x])^2] = \notag \\
    &= f(E[x]) + \frac12 f''(E[x]) \sigma_x^2. \tag{1} \label{eq:mprop}
    \f}
    
    Had we done a first-order error propagation, the formula would have been
    \f$ E[y] \approx f(E[x]) \f$, which intuitively reads «just compute the
    function at the mean of \f$ x \f$». The new term reads «increase the
    mean is the function bends upward, decrease if the function bends downward»,
    which actually makes sense.
    
    Second thing: in general, in the formula for \f$ E[y^n] \f$, the highest
    \f$ x \f$ moment that appears is \f$ E[x^{2n}] \f$.
    
    Estimates
    ---------
    
    We have defined second order moment propagation, we know what it means, we
    could write down all the formulae. So we can start using the class, right?
    Wrong.
    
    We first need to introduce the concept of _estimate_. Let \f$ \theta \f$ be
    the "true value" of an unknown quantity we have to measure. Tipically the
    output of the measure is a random variable \f$ \hat\theta \f$ that in some
    sense should “estimate” \f$ \theta \f$. Conventionally, \f$ \hat\theta \f$
    is considered a good estimate if these two properties hold:
    
    1) \f$ E[\hat\theta] = \theta \f$, and
    
    2) the shape of the distribution of \f$ \hat\theta \f$ does not depend, or
    depends “not too much” on the value of \f$ \theta \f$.
    
    Property (1) is called _unbiasedness_, because the _bias_ of \f$ \hat\theta
    \f$ is defined as \f$ E[\hat\theta] - \theta \f$. This property may or may
    not make sense to you at first sight, but please trust me on its usefulness
    as conventional choice since a complete explanation would be a bit long.
    
    Property (2) has something to do with confidence intervals. You may have
    encountered something like “the 95 % confidence interval is given by
    \f$ \pm 2 \f$ standard deviations”, well, for something like that to have a
    chance to be true, property (2) must hold.
    
    A simple example in which property (2) does not hold is when you have a
    constant relative error, which means the standard deviation of the
    distribution of \f$ \hat\theta \f$ is proportional to \f$ \theta \f$, so
    the shape *does* change significatively with \f$ \theta \f$. (In this case
    it may be a good idea to use an estimate of \f$ \log\theta \f$.)
    
    Now, let's suppose you have measured \f$ \theta \f$, in the sense outlined
    above, and that you are interested in measuring the transformed quantity
    \f$ \theta' = f(\theta) \f$. We have to obtain a measure \f$ \hat\theta'
    \f$ that fulfills the properties, starting from \f$ \hat\theta \f$. The
    first thing that comes to mind is just using \f$ \hat\theta' =
    f(\hat\theta) \f$. Let's compute the bias of this quantity. We have to
    compute \f$ E[f(\hat\theta)] \f$. If we do the calculation to second order,
    we can just apply the formula we found before:
    
    \f{align*}{
    E[\hat\theta' = f(\hat\theta)]
    &\approx f(E[\hat\theta]) + \frac12 f''(E[\hat\theta]) \sigma_{\hat\theta}^2 = \\
    &= f(\theta) + \frac12 f''(\theta) \sigma_{\hat\theta}^2 = \\
    &= \theta' + \frac12 f''(\theta) \sigma_{\hat\theta}^2
    \f}
    
    So this choice of \f$ \hat\theta' \f$ is biased. Considering that
    \f$ f''(\hat\theta) = f''(\theta) + O(\hat\theta - \theta) \f$, we can just
    subtract the bias to obtain an unbiased (to second order) estimate:
    
    \f{align}{
    \hat\theta' = f(\hat\theta) - \frac12 f''(\hat\theta) \sigma_{\hat\theta}^2.
    \tag{2}
    \label{eq:eprop}
    \f}
    
    The last formula we wrote is quite important. It differs from the
    propagation formula in the sign of the second term. So, when propagating
    a measurement through calculations we actually have to invert the sign of
    the original formula... What's going on?
    
    Let's make a simple concrete example. Let \f$ \vec v = v_x \hat x \f$ be
    the velocity of a particle moving on a 1D axis. Let's suppose that the
    particle is still, i.e. \f$ v_x = 0 \f$. We can measure \f$ v_x \f$
    obtaining estimates \f$ \hat v_x \f$ that are distributed normally with
    standard deviation \f$ \sigma_{\hat v_x} = 1 \f$ (arbitrary units), and of
    course unbiased: \f$ E[\hat v_x] = v_x = 0 \f$.
    
    Suppose that, given a masurement of the velocity, we want to compute the
    kinetical energy \f$ E = 1/2 m v_x^2 \f$. The particle is still, so we
    want to obtain \f$ E = 0 \f$ on average if we repeat the measurement many
    times. If we compute \f$ \hat E = 1/2 m \hat v_x^2 \f$, we have always
    \f$ \hat E \ge 0 \f$, so \f$ E[\hat E] \f$ will be a positive number, which
    means \f$ \hat E \f$ will be biased.
    
    We can compute the bias with formula \f$ \eqref{eq:mprop} \f$, which in this
    case is exact because the formula is quadratic:
    
    \f{align*}{
    E \left[ \frac12 m \hat v_x^2 \right]
    &= \frac12 m v_x^2 + \frac12 m \sigma_{\hat v_x}^2 = \\
    &= \frac12 m.
    \f}
    
    To get an unbiased estimate, we just subtract the bias:
    
    \f[
    \hat E_{\text{unbiased}} = \frac12 m \hat v_x^2 - \frac12 m.
    \f]
    
    Note that we could have obtained the final result by applying directly
    formula \f$ \eqref{eq:eprop} \f$, even without knowing the true value
    \f$ v_x = 0 \f$.
    
    Observation: an unbiased estimate of a positive quantity can yield negative
    values.
    
    Conclusions
    -----------
    
    We have introduced two different formulas \f$ \eqref{eq:mprop} \f$ and \f$
    \eqref{eq:eprop} \f$ for the propagation of the "mean" of a variable. (We
    have not said what happens for __higher moments__: currently __only the
    formulae for moment propagation__ and not those for unbiased moment
    estimation __are implemented__.) Depending on what you are doing, you have
    to pick one.
    
    If you have a variable of which you know the moments, and you want to
    see how the mean changes if you transform the variable, use formula
    \f$ \eqref{eq:mprop} \f$.
    
    If you have an unbiased estimate of a quantity, and you want to obtain an
    unbiased estimate of a trasformation of the quantity, use formula
    \f$ \eqref{eq:eprop} \f$.
    
    These two choices are implemented in the template parameter `prop`: use
    `UReal2<..., Prop::mean>` for \f$ \eqref{eq:mprop} \f$ and
    `UReal2<..., Prop::est>` for \f$ \eqref{eq:eprop} \f$.
    
    Basics
    ======
    
    The template parameter `Real` is the numerical type used, and is aliased to
    the member type `real_type`; the parameter `prop` is the kind of propagation
    to use, `Prop::mean` means moment propagation and `Prop::est` unbiased
    estimate propagation.
    
    The aliases `UReal2E<Real>`, `UReal2M<Real>`, `udouble2e` and `udouble2m`
    are provided for convenience.
    
    Initialization
    --------------
    
    The simple way of initializing a variable is using one of the functions in
    `distr.hpp`. Otherwise, you have to manually specify the first 8 central
    moments.
    
    ~~~cpp
    #include <uncertainties/ureal2.hpp>
    #include <uncertainties/distr.hpp>
    ...
    namespace unc = uncertainties;
    unc::udouble2e x(0, 1, {0, 3, 0, 15, 0, 105}); // normal distribution
    // the first two numbers are mean and standard deviation
    // the array of 6 numbers are the standardized central moments 3 to 8
    x = unc::distr::normal<unc::udouble2e>(0, 1); // the same
    ~~~
    
    Accessing the moments
    ---------------------
    
    The member functions `s`, `skew` and `kurt` compute the standard deviation,
    the skewness and the kurtosis respectively. The member function `n` computes
    the "mean", either in the sense of formula \f$ \eqref{eq:mprop} \f$ or
    \f$ \eqref{eq:eprop} \f$ depending on the template parameter `prop`.
    The member function `first_order_n` computes the mean to first order, like
    `UReal` would do.
    
    The member function `m(n)` computes the `n`th central moment (non
    standardized). `m(1)` computes the correction term that is used to get the
    mean, but the sign is always as if `prop == Prop::mean`. So
    `x.n() == x.first_order_n() + x.m(1)` if `prop == Prop::mean`, and
    `x.n() == x.first_order_n() - x.m(1)` if `prop == Prop::est`.
    
    Printing
    --------
    
    An `UReal2` can be formatted to string and output to a stream. You have to
    include the header `io.hpp`.
    
    ~~~cpp
    #include <uncertainties/io.hpp>
    ...
    unc::udouble2e x = unc::distr::normal<unc::udouble2e>(13, 0.4);
    std::cout << x << "\n"; // will print "13.0 ± 0.4"
    std::cout << x.format(3) << "\n"; // "13.000 ± 0.400"
    std::cout << format(x, 3) << "\n"; // the same
    ~~~
    
    See the function `format` in `io.hpp` for details.
    
    Functions
    ---------
    
    Standard mathematical functions are defined in `math.hpp`. User defined
    functions can be created using the utilities in `functions.hpp`.
    
    Dependent vs independent variables
    ==================================
    
    When a `UReal2` is initialized with mean and standard deviation, it is an
    _independent variable_. Conversely, values obtained through operations on
    `UReal2`s are _dependent variables_. Dependent variables store internally
    information on all the independent variables that entered the computation.
    The (in)dependency of a variable can be tested with the member function
    `isindep`:
    
    ~~~cpp
    unc::udouble2e x = unc::distr::normal<unc::udouble2e>(1, 0.1);
    bool ind = x.isindep(); // true
    unc::udouble2e y = unc::distr::normal<unc::udouble2e>(0.5, 0.2);
    unc::udouble2e z = x + y;
    ind = z.isindep(); // false
    ~~~
    
    A value uniquely identifying an independent variable can be obtained with
    `indepid`:
    
    ~~~cpp
    unc::Id id = x.indepid(); // some value
    id = y.indepid(); // a value different from the one of x
    id = z.indepid(); // returns unc::invalid_id
    ~~~
    
    The independent variable id is the same as for the class `UReal`. There will
    not be an `UReal2` and a `UReal` with the same id.
    
    If a computation involves only one variable, the result still classifies as
    independent, in particular a copy of an independent variable is still
    independent:
    
    ~~~cpp
    z = x + x;
    id = z.isindep(); // true
    y = x;
    x.isindep(); // true
    y.isindep(); // true
    ~~~
    
    Implementing functions
    ======================
    
    The header `math.hpp` defines standard mathematical functions on `UReal2`s.
    Other functions can be easily added with the utilities in `functions.hpp`.
    
    If that is not sufficient, you can use directly the friend functions
    `unary` and `binary`. They require you to compute manually the numbers
    required.
    
    Example re-implementing sin:
    
    ~~~cpp
    #include <cmath>
    ...
    unc::udouble2e usin(unc::udouble2e &x) {
        double mean = sin(x.first_order_n());
        double dsindx = cos(x.first_order_n());
        double ddsindxdx = -sin(x.first_order_n());
        // note: we use `first_order_n()`, not `n()`
        return unary(x, mean, dsindx, ddsindxdx);
    }
    ~~~
    
    Example re-implementing multiplication:
    
    ~~~cpp
    unc::udouble2e umult(unc::udouble2e &x, unc::udouble2e &y) {
        double mean = x.first_order_n() * y.first_order_n();
        double dmdx = y.first_order_n();
        double dmdy = x.first_order_n();
        double ddmdxdx = 0;
        double ddmdydy = 0;
        double ddmdxdy = 1;
        return binary(x, y, mean, dmdx, dmdy, ddmdxdx, ddmdydy, ddmdxdy);
    }
    ~~~
    
    */
    template<typename Real, Prop prop>
    class UReal2 {
    private:
        using HessGrad = internal::HessGrad<Real>;
        using Diag = typename HessGrad::Diag;
        using ConstDiagIt = typename HessGrad::ConstDiagIt;
        using ConstTriIt = typename HessGrad::ConstTriIt;
        using ConstHessIt = typename HessGrad::ConstHessIt;
        
        // variables
        HessGrad hg;
        Real mu {0};

    public:
        /*!
        \brief Numerical type used.
        */
        using real_type = Real;
        
        /*!
        \brief Propagation mode, `Prop::mean` or `Prop::est`.
        */
        static constexpr Prop prop_mode = prop;
        
        /*!
        \brief Construct a variable given central moments.
        
        `n` is the mean. `moments` is the array of central moments from the
        second to the eighth.
        
        Not all sequences of numbers are valid moments for a distribution. The
        given moments are checked and an exception is thrown if they are not
        valid. The checking is sensitive to floating point errors, so you can
        set a threshold with `check_moments_threshold`. The default value of 0
        is a safe choice. Since the checking is computationally expensive,
        you can disable checking by setting `check_moments_threshold` to a
        negative value.
        
        \throw std::invalid_argument if the moments are not valid.
        */
        UReal2(const Real &n,
               const std::array<Real, 7> &moments,
               const Real &check_moments_threshold=0):
        mu {n} {
            if (not (check_moments_threshold < 0)) {
                internal::check_moments_throw(moments, check_moments_threshold);
            }
            const Id id = ++internal::last_id;
            Diag &diag = this->hg.diag(id);
            using std::sqrt;
            diag.grad = sqrt(moments[0]);
            diag.hhess = 0;
            if (diag.grad > 0) {
                Real sn = moments[0];
                for (int i = 0; i < 6; ++i) {
                    sn *= diag.grad;
                    diag.mom[i] = moments[i + 1] / sn;
                }
            } else {
                std::fill(diag.mom.begin(), diag.mom.end(), 0);
            }
        }
        
        /*!
        \brief Construct a variable given standardized central moments.
        
        `n` is the mean, `s` the standard deviation and `std_moments` the
        standardized central moments from third to eighth.
        
        \throw std::invalid_argument if `s < 0` or if the moments are not valid.
        */
        UReal2(const Real &n, const Real &s,
               const std::array<Real, 6> &std_moments,
               const Real &check_moments_threshold=0):
        mu {n} {
            if (s < 0) {
                std::ostringstream ss;
                ss << "uncertainties::UReal2::UReal2: s = " << s << " < 0";
                throw std::invalid_argument(ss.str());
            }
            if (not (check_moments_threshold < 0)) {
                internal::check_moments_throw(std_moments, check_moments_threshold);
            }
            const Id id = ++internal::last_id;
            Diag &diag = this->hg.diag(id);
            diag.grad = s;
            diag.hhess = 0;
            diag.mom = internal::Moments<Real>(std_moments);
        }
        
        /*!
        \brief Construct a variable with mean `n` and all central moments 0.
        */
        inline UReal2(const Real &n) noexcept:
        mu {n} {
            ;
        }
        
        /*!
        \brief Construct a variable with all moments zero.
        */
        inline UReal2() noexcept {
            ;
        }
        
        /*!
        \brief Is the variable independent?
        */
        inline bool isindep() const noexcept {
            return this->hg.size() <= 1;
        }

        /*!
        \brief Return the independent variable id, or `invalid_id` if it is not
        independent.
        */
        Id indepid() const noexcept {
            if (not this->isindep()) {
                return invalid_id;
            } else if (this->hg.size() == 1) {
                return this->hg.cdbegin()->first;
            } else {
                return 0;
            }
        }
        
        /*!
        \brief Return the mean propagated to first order.
        
        It is the same result that would be computed by `UReal`, and is just
        the result of the computation carried on with normal numbers.
        */
        inline const Real &first_order_n() const noexcept {
            return this->mu;
        }
        
        /*!
        \brief Compute the mean if `prop == Prop::mean`, the unbiased estimate
        if `prop == Prop::est`.
        */
        Real n() const noexcept {
            return this->mu + internal::propsign(prop) * this->m(1);
        }
        
        /*!
        \brief Compute the stadard deviation.
        */
        Real s() const noexcept {
            using std::sqrt;
            return sqrt(this->m(2));
        }
        
        /*!
        \brief Compute the skewness.
        */
        Real skew() const noexcept {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        /*!
        \brief Compute the kurtosis.
        */
        Real kurt() const noexcept {
            const Real s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        /*!
        \brief Compute the (non standardized) `n`th central moment.
        
        `n` must be 1, 2, 3, or 4. `m(1)` is actually the correction that
        is applied to `first_order_n()` to obtain `n()`, in this way:
        `x.n() == x.first_order_n() + x.m(1)` if `prop == Prop::mean`, and
        `x.n() == x.first_order_n() - x.m(1)` if `prop == Prop::est`.
        */
        Real m(const int n) const {
            assert(n >= 1 and n <= 4);
            std::array<Real, 4> mom;
            for (int i = 0; i < n; ++i) {
                internal::update_mom(mom, i, this->hg);
            }
            return mom[n - 1];
        }

        /*!
        \brief Format the variable to string.
        
        Just calls `format(*this, args...)`. See `format.hpp`.
        */
        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        template<typename OtherReal, Prop other_prop>
        friend class UReal2;
        
        /*!
        \brief Explicit cast to `UReal2` with different numerical type and
        propagation mode.
        */
        template<typename AnyReal, Prop any_prop>
        explicit UReal2(const UReal2<AnyReal, any_prop> &x):
        hg {x.hg}, mu {static_cast<Real>(x.mu)} {
            ;
        }
        
        /*!
        \brief Implicit cast to `UReal2` with different numerical type and
        same propagation mode.
        
        The implicit cast will work only if the implicit cast from `Real` to
        `OtherReal` is safe.
        */
        template<typename OtherReal>
        operator UReal2<OtherReal, prop>() const {
            UReal2<OtherReal, prop> x;
            x.hg = this->hg;
            x.mu = OtherReal {this->mu};
            return x;
        }
        
        /*!
        \brief Compute the variance.
        */
        inline friend Real var(const UReal2<Real, prop> &x) noexcept {
            return x.m(2);
        }
        
        /*!
        \brief Compute the covariance.
        */
        inline friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) noexcept {
            return internal::compute_c2(x.hg, y.hg) - x.m(1) * y.m(1);
        }
        
        /*!
        \brief Compute the correlation.
        */
        inline friend Real corr(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) noexcept {
            return cov(x, y) / (x.s() * y.s());
        }
        
        /*!
        \brief Compute the first derivative with respect to `x`.
        
        \throw std::invalid_argument if `x` is not independent or if `x` does
        not contain uncertainty information, which happens if `x` is the
        result of the default constructor or the one-argument constructor.
        */
        Real grad(const UReal2<Real, prop> &x) const {
            if (x.hg.size() != 1) {
                std::ostringstream ss;
                ss << "uncertainties::UReal2::grad: ";
                ss << "variable is not independent and nontrivial";
                throw std::invalid_argument(ss.str());
            }
            const ConstDiagIt it = x.hg.cdbegin();
            return this->hg.diag_get(it->first).grad / it->second.grad;
        }
        
        /*!
        \brief Compute the second derivative with respect to `x`, `y`.
        
        \throw std::invalid_argument if `x` or `y` are not independent or do not
        have uncertainty information, or if `x` and `y` originate from the same
        independent variable but with different dependency.
        */
        Real hess(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) const {
            if (x.hg.size() != 1 or y.hg.size() != 1) {
                std::ostringstream ss;
                ss << "uncertainties::UReal2::hess: ";
                ss << "variables are not independent and nontrivial";
                throw std::invalid_argument(ss.str());
            }
            const Id idx = x.hg.cdbegin()->first;
            const Id idy = y.hg.cdbegin()->first;
            if (idx != idy) {
                const Real &xgrad = x.hg.cdbegin()->second.grad;
                const Real &ygrad = y.hg.cdbegin()->second.grad;
                return 2 * this->hg.hhess_get(idx, idy) / (xgrad * ygrad);
            } else {
                const Id id = idx;
                const Real &dxdv = x.hg.cdbegin()->second.grad;
                const Real &dydv = y.hg.cdbegin()->second.grad;
                const Real ddxdvdv = 2 * x.hg.cdbegin()->second.hhess;
                const Real ddydvdv = 2 * y.hg.cdbegin()->second.hhess;
                if (dxdv != dydv or ddxdvdv != ddydvdv) {
                    std::ostringstream ss;
                    ss << "uncertainties::UReal2::hess: ";
                    ss << "variables have different dependency on the same id";
                    throw std::invalid_argument(ss.str());
                }
                const Real &dfdv = this->hg.diag_get(id).grad;
                const Real ddfdvdv = 2 * this->hg.hhess_get(id, id);
                return (ddfdvdv - (dfdv / dxdv) * ddxdvdv) / (dxdv * dxdv);
            }
        }
        
        /*!
        \brief Compute a one argument function.
        
        `fx`, `dfdx`, `ddfdxdx` must be respectively the function, its first
        derivative and its second derivative computed at `x.first_order_n()`.
        */
        friend UReal2<Real, prop> unary(
            const UReal2<Real, prop> &x,
            const Real &fx, const Real &dfdx, const Real &ddfdxdx
        ) {
            UReal2<Real, prop> result;
            result.mu = fx;
            
            const Real hddfdxdx = ddfdxdx / 2;
            
            const ConstHessIt end = x.hg.chend();
            ConstHessIt it = x.hg.chbegin(hddfdxdx == 0);
            for (; it != end; ++it) {
                Real &hhess = result.hg.hhess(it.id1(), it.id2());
                hhess = hddfdxdx * it.diag1().grad * it.diag2().grad;
                hhess += dfdx * (*it);
                if (it.id1() == it.id2()) {
                    Diag &diag = result.hg.diag(it.id1());
                    diag.mom = it.diag1().mom;
                    diag.grad = dfdx * it.diag1().grad;
                }
            }
            
            return result;
        }        
        
        /*!
        \brief Compute a two argument function.
        
        `fxy`, `dfdx`, `dfdy`, `ddfdxdx`, `ddfdydy`, `ddfdxdy` must the
        function, its first derivatives and its second derivatives computed at
        `x.first_order_n()`, `y.first_order_n()`.
        */
        friend UReal2<Real, prop> binary(
            const UReal2<Real, prop> &x, const UReal2<Real, prop> &y,
            const Real &fxy,
            const Real &dfdx, const Real &dfdy,
            const Real &ddfdxdx, const Real &ddfdydy, const Real &ddfdxdy
        ) {
            UReal2<Real, prop> result;
            result.mu = fxy;

            constexpr Id maxid = std::numeric_limits<Id>::max();
            const std::pair<Id, Id> pmaxid {maxid, maxid};

            const Real hddfdxdx = ddfdxdx / 2;
            const Real hddfdydy = ddfdydy / 2;
            const Real hddfdxdy = ddfdxdy / 2;
        
            ConstHessIt itx = x.hg.chbegin(hddfdxdx == 0);
            const ConstHessIt xend = x.hg.chend();
            ConstHessIt ity = y.hg.chbegin(hddfdydy == 0);
            const ConstHessIt yend = y.hg.chend();
            
            while (itx != xend or ity != yend) {
                const std::pair<Id, Id> idx = itx != xend ? std::make_pair(itx.id1(), itx.id2()) : pmaxid;
                const std::pair<Id, Id> idy = ity != yend ? std::make_pair(ity.id1(), ity.id2()) : pmaxid;
                
                Real *hhess;
                Diag *diag;
                
                if (idx <= idy) {
                    hhess = &result.hg.hhess(idx.first, idx.second);
                    *hhess += hddfdxdx * itx.diag1().grad * itx.diag2().grad;
                    *hhess += dfdx * (*itx);
                    if (idx.first == idx.second) {
                        diag = &result.hg.diag(idx.first);
                        diag->mom = itx.diag1().mom;
                        diag->grad += dfdx * itx.diag1().grad;
                    }
                }
                
                if (idy <= idx) {
                    if (idx != idy) {
                        hhess = &result.hg.hhess(idy.first, idy.second);
                        if (idy.first == idy.second) {
                            diag = &result.hg.diag(idy.first);
                            diag->mom = ity.diag1().mom;
                        }
                    }
                    *hhess += hddfdydy * ity.diag1().grad * ity.diag2().grad;
                    *hhess += dfdy * (*ity);
                    if (idy.first == idy.second) {
                        diag->grad += dfdy * ity.diag1().grad;
                    }
                }
                
                if (idx <= idy) ++itx;
                if (idy <= idx) ++ity;
            }
            
            if (hddfdxdy != 0) {
                const ConstDiagIt xend = x.hg.cdend();
                const ConstDiagIt yend = y.hg.cdend();
                for (ConstDiagIt itx = x.hg.cdbegin(); itx != xend; ++itx) {
                    const Id idx = itx->first;
                    const Diag &dx = itx->second;
                    for (ConstDiagIt ity = y.hg.cdbegin(); ity != yend; ++ity) {
                        const Id idy = ity->first;
                        const Diag &dy = ity->second;
                        const Id minid = std::min(idx, idy);
                        const Id maxid = std::max(idx, idy);
                        const Real addend = hddfdxdy * dx.grad * dy.grad;
                        if (minid != maxid) {
                            result.hg.tri(minid, maxid) += addend;
                        } else {
                            result.hg.diag(minid).hhess += 2 * addend;
                        }
                    }
                }
            }

            return result;
        }
        
        /*!
        \brief Compute a two argument function, storing the result in `x`.
        
        `fxy`, `dfdx`, `dfdy`, `ddfdxdx`, `ddfdydy`, `ddfdxdy` must the
        function, its first derivatives and its second derivatives computed at
        `x.first_order_n()`, `y.first_order_n()`.
        */
        friend void binary_assign(
            UReal2<Real, prop> &x, const UReal2<Real, prop> &y,
            const Real &fxy,
            const Real &dfdx, const Real &dfdy,
            const Real &ddfdxdx, const Real &ddfdydy, const Real &ddfdxdy
        ) {
            x.mu = fxy;

            constexpr Id maxid = std::numeric_limits<Id>::max();
            const std::pair<Id, Id> pmaxid {maxid, maxid};

            const Real hddfdxdx = ddfdxdx / 2;
            const Real hddfdydy = ddfdydy / 2;
            const Real hddfdxdy = ddfdxdy / 2;
        
            ConstHessIt itx = x.hg.chbegin(hddfdxdx == 0);
            const ConstHessIt xend = x.hg.chend();
            ConstHessIt ity = y.hg.chbegin(hddfdydy == 0);
            const ConstHessIt yend = y.hg.chend();
            
            while (itx != xend or ity != yend) {
                const std::pair<Id, Id> idx = itx != xend ? std::make_pair(itx.id1(), itx.id2()) : pmaxid;
                const std::pair<Id, Id> idy = ity != yend ? std::make_pair(ity.id1(), ity.id2()) : pmaxid;
                
                Real *hhess;
                Diag *diag;
                
                if (idx <= idy) {
                    hhess = &x.hg.hhess(idx.first, idx.second);
                    *hhess += hddfdxdx * itx.diag1().grad * itx.diag2().grad;
                    *hhess += dfdx * (*itx);
                    if (idx.first == idx.second) {
                        diag = &x.hg.diag(idx.first);
                        diag->mom = itx.diag1().mom;
                        diag->grad += dfdx * itx.diag1().grad;
                    }
                }
                
                if (idy <= idx) {
                    if (idx != idy) {
                        hhess = &x.hg.hhess(idy.first, idy.second);
                        if (idy.first == idy.second) {
                            diag = &x.hg.diag(idy.first);
                            diag->mom = ity.diag1().mom;
                        }
                    }
                    *hhess += hddfdydy * ity.diag1().grad * ity.diag2().grad;
                    *hhess += dfdy * (*ity);
                    if (idy.first == idy.second) {
                        diag->grad += dfdy * ity.diag1().grad;
                    }
                }
                
                if (idx <= idy) ++itx;
                if (idy <= idx) ++ity;
            }
            
            // I shall consider the following code a very inefficient patch.
            if (hddfdxdy != 0) {
                const ConstDiagIt end = x.hg.cdend();
                ConstDiagIt iti, itj;
                for (iti = x.hg.cdbegin(); iti != end; ++iti) {
                    for (itj = iti; itj != end; ++itj) {
                        const Id idi = iti->first;
                        const Id idj = itj->first;
                        const Real &dix = x.hg.diag_get(idi).grad;
                        const Real &djx = x.hg.diag_get(idj).grad;
                        const Real &diy = y.hg.diag_get(idi).grad;
                        const Real &djy = y.hg.diag_get(idj).grad;
                        x.hg.hhess(idi, idj) += hddfdxdy * (dix * djy + djx * diy);
                    }
                }
            }
        }
        
        friend inline const UReal2<Real, prop> &
        operator+(const UReal2<Real, prop> &x) noexcept {
            return x;
        }
        
        friend UReal2<Real, prop>
        operator-(const UReal2<Real, prop> &x) {
            return unary(x, -x.first_order_n(), -1, 0);
        }
    
        friend UReal2<Real, prop>
        operator+(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            return binary(x, y,
                          x.first_order_n() + y.first_order_n(),
                          1, 1, 0, 0, 0);
        }
    
        friend UReal2<Real, prop>
        operator-(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            return binary(x, y,
                          x.first_order_n() - y.first_order_n(),
                          1, -1, 0, 0, 0);
        }
    
        friend UReal2<Real, prop>
        operator*(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            const Real xn = x.first_order_n();
            const Real yn = y.first_order_n();
            return binary(x, y,
                          xn * yn,
                          yn, xn,
                          0, 0, 1);
        }

        friend UReal2<Real, prop>
        operator/(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            const Real xn = x.first_order_n();
            const Real yn = y.first_order_n();
            const Real invy = 1 / yn;
            const Real invy2 = invy * invy;
            return binary(x, y,
                          xn / yn,
                          invy, -xn * invy2,
                          0, 2 * xn * invy2 * invy, -invy2);
        }

        template<typename Predicate>
        friend bool all(const UReal2<Real, prop> &x, Predicate p) {
            if (not p(x.mu)) {
                return false;
            }
            const ConstDiagIt dend = x.hg.cdend();
            for (ConstDiagIt it = x.hg.cdbegin(); it != dend; ++it) {
                const Diag &diag = it->second;
                if (not p(diag.grad) or not p(diag.hhess)) {
                    return false;
                }
            }
            const ConstTriIt tend = x.hg.ctend();
            for (ConstTriIt it = x.hg.ctbegin(true); it != tend; ++it) {
                if (not p(*it)) {
                    return false;
                }
            }
            return true;
        }
    };
    
    template<typename Real>
    using UReal2E = UReal2<Real, Prop::est>;
    
    template<typename Real>
    using UReal2M = UReal2<Real, Prop::mean>;
    
    using udouble2e = UReal2E<double>;
    using udouble2m = UReal2M<double>;
    
    using ufloat2e = UReal2E<float>;
    using ufloat2m = UReal2M<float>;
    
    /*!
    \brief Return `x.n()`.
    */
    template<typename Real, Prop prop>
    inline Real nom(const UReal2<Real, prop> &x) noexcept {
        return x.n();
    }

    /*!
    \brief Return `x.s()`.
    */
    template<typename Real, Prop prop>
    inline Real sdev(const UReal2<Real, prop> &x) noexcept {
        return x.s();
    }

#ifdef UNCERTAINTIES_EXTERN_UDOUBLE2
    extern template class UReal2<double, Prop::est>;
    extern template class UReal2<double, Prop::mean>;
#endif
}

#endif /* end of include guard: UNCERTAINTIES_UREAL2_HPP_4F22829C */
