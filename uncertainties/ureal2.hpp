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
    under the hood. Just remember the following things.
    
    First thing: let's write the formula for the mean of \f$ y \f$ by
    translating back everything:
    
    \f{align*}{
    E[y] &\approx f(E[x]) + \frac12 f''(E[x]) E[(x-E[x])^2] = \\
    &= f(E[x]) + \frac12 f''(E[x]) \sigma_x^2.
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
    
    \f{align*}{
    \hat\theta' = f(\hat\theta) - \frac12 f''(\hat\theta) \sigma_{\hat\theta}^2.
    \f}
    
    The last formula we wrote is quite important. It differs from the
    propagation formula in the sign of the second term. So, when propagating
    a measurement through calculations we actually have to invert the sign of
    the original formula... What's going on?
    
    Let's make a simple example.
    
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
        std::array<Real, 4> mom {0, 0, 0, 0};
        std::array<bool, 4> mom_to_compute {false, false, false, false};
        
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
        
    public:
        using real_type = Real;
        static constexpr Prop prop_mode = prop;
        
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
            std::copy(moments.begin(), moments.begin() + 3, this->mom.begin() + 1);
        }
        
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
            Real sn = s * s;
            this->mom[1] = sn;
            for (int i = 2; i < 4; ++i) {
                sn *= s;
                this->mom[i] = std_moments[i - 2] * sn;
            }
        }
        
        UReal2(const Real &n):
        mu {n} {
            ;
        }

        UReal2() {
            ;
        }
        
        inline bool isindep() const noexcept {
            return this->hg.size() <= 1;
        }

        Id indepid() const noexcept {
            if (not this->isindep()) {
                return invalid_id;
            } else if (this->hg.size() == 1) {
                return this->hg.cdbegin()->first;
            } else {
                return 0;
            }
        }

        inline const Real &first_order_n() const noexcept {
            return this->mu;
        }
        
        Real n() noexcept  {
            return this->mu + internal::propsign(prop) * this->m(1);
        }

        Real n() const noexcept {
            return this->mu + internal::propsign(prop) * this->m(1);
        }
        
        Real s() noexcept {
            using std::sqrt;
            return sqrt(this->m(2));
        }

        Real s() const noexcept {
            using std::sqrt;
            return sqrt(this->m(2));
        }
        
        Real skew() noexcept {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        Real skew() const noexcept {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        Real kurt() noexcept {
            const Real &s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        Real kurt() const noexcept {
            const Real s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        const Real &m(const int n) {
            assert(n >= 1 and n <= 4);
            for (int i = 0; i < n; ++i) {
                if (this->mom_to_compute[i]) {
                    internal::update_mom(this->mom, i, this->hg);
                    this->mom_to_compute[i] = false;
                }
            }
            return this->mom[n - 1];
        }
        
        Real m(const int n) const {
            assert(n >= 1 and n <= 4);
            if (this->mom_to_compute[n - 1]) {
                std::array<Real, 4> mom;
                for (int i = 0; i < n; ++i) {
                    if (this->mom_to_compute[i]) {
                        internal::update_mom(mom, i, this->hg);
                    } else {
                        mom[i] = this->mom[i];
                    }
                }
                return mom[n - 1];
            } else {
                return this->mom[n - 1];
            }
        }

        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        template<typename OtherReal, Prop other_prop>
        friend class UReal2;
        
        // explicit cast
        template<typename AnyReal, Prop any_prop>
        explicit UReal2(const UReal2<AnyReal, any_prop> &x):
        hg {x.hg}, mu {static_cast<Real>(x.mu)}, mom_to_compute {x.mom_to_compute} {
            std::copy(x.mom.begin(), x.mom.end(), this->mom.begin());
        }
        
        // implicit cast if prop is the same and there is safe cast
        // OtherReal -> Real
        template<typename OtherReal>
        operator UReal2<OtherReal, prop>() const {
            UReal2<OtherReal, prop> x;
            x.hg = this->hg;
            x.mu = OtherReal {this->mu};
            x.mom_to_compute = this->mom_to_compute;
            std::copy(this->mom.begin(), this->mom.end(), x.mom.begin());
            return x;
        }
        
        inline friend Real var(const UReal2<Real, prop> &x) noexcept {
            return x.m(2);
        }
        
        inline friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) noexcept {
            return internal::compute_c2(x.hg, y.hg) - x.m(1) * y.m(1);
        }
        
        Real _grad(const UReal2<Real, prop> &x) const {
            assert(x.hg.size() == 1);
            const ConstDiagIt it = x.hg.cdbegin();
            return this->hg.diag_get(it->first).grad / it->second.grad;
        }
        
        Real _hess(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) const {
            assert(x.hg.size() == 1);
            assert(y.hg.size() == 1);
            const Id idx = x.hg.cdbegin()->first;
            const Id idy = y.hg.cdbegin()->first;
            if (idx != idy) {
                const Real xgrad = x.hg.cdbegin()->second.grad;
                const Real ygrad = y.hg.cdbegin()->second.grad;
                return 2 * this->hg.hhess_get(idx, idy) / (xgrad * ygrad);
            } else {
                const Id id = idx;
                const Real dxdv = x.hg.cdbegin()->second.grad;
                const Real dydv = y.hg.cdbegin()->second.grad;
                const Real ddxdvdv = 2 * x.hg.cdbegin()->second.hhess;
                const Real ddydvdv = 2 * y.hg.cdbegin()->second.hhess;
                assert(dxdv == dydv and ddxdvdv == ddydvdv);
                const Real dfdv = this->hg.diag_get(id).grad;
                const Real ddfdvdv = 2 * this->hg.hhess_get(id, id);
                return (ddfdvdv - (dfdv / dxdv) * ddxdvdv) / (dxdv * dxdv);
            }
        }
        
        friend UReal2<Real, prop> unary(
            const UReal2<Real, prop>x,
            const Real &fx, const Real &dfdx, const Real &ddfdxdx
        ) {
            UReal2<Real, prop> result;
            result.mu = fx;
            for (bool &b : result.mom_to_compute) {
                b = true;
            }
            
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
        
        friend UReal2<Real, prop> binary(
            const UReal2<Real, prop> &x, const UReal2<Real, prop> &y,
            const Real &fxy,
            const Real &dfdx, const Real &dfdy,
            const Real &ddfdxdx, const Real &ddfdydy, const Real &ddfdxdy
        ) {
            UReal2<Real, prop> result;
            result.mu = fxy;
            for (bool &b : result.mom_to_compute) {
                b = true;
            }

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
            
            // I shall consider the following code a very inefficient patch.
            if (hddfdxdy != 0) {
                const ConstDiagIt end = result.hg.cdend();
                ConstDiagIt iti, itj;
                for (iti = result.hg.cdbegin(); iti != end; ++iti) {
                    for (itj = iti; itj != end; ++itj) {
                        const Id idi = iti->first;
                        const Id idj = itj->first;
                        const Real dix = x.hg.diag_get(idi).grad;
                        const Real djx = x.hg.diag_get(idj).grad;
                        const Real diy = y.hg.diag_get(idi).grad;
                        const Real djy = y.hg.diag_get(idj).grad;
                        result.hg.hhess(idi, idj) += hddfdxdy * (dix * djy + djx * diy);
                    }
                }
            }

            return result;
        }
        
        inline const UReal2<Real, prop> &operator+() const noexcept {
            return *this;
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
        
        friend inline bool isfinite(const UReal2<Real, prop> &x) noexcept {
            using std::isfinite;
            return all(x, [](const Real &n) { return isfinite(n); });
        }
        
        friend inline bool isnormal(const UReal2<Real, prop> &x) noexcept {
            using std::isnormal;
            return all(x, [](const Real &n) { return isnormal(n); });
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
    
    template<typename Real, Prop prop>
    inline Real nom(const UReal2<Real, prop> &x) noexcept {
        return x.n();
    }

    template<typename Real, Prop prop>
    inline Real nom(UReal2<Real, prop> &x) noexcept {
        return x.n();
    }
    
    template<typename Real, Prop prop>
    inline Real sdev(const UReal2<Real, prop> &x) noexcept {
        return x.s();
    }

    template<typename Real, Prop prop>
    inline Real sdev(UReal2<Real, prop> &x) noexcept {
        return x.s();
    }

#ifdef UNCERTAINTIES_EXTERN_UDOUBLE2
    extern template class UReal2<double, Prop::est>;
    extern template class UReal2<double, Prop::mean>;
#endif
}

#endif /* end of include guard: UNCERTAINTIES_UREAL2_HPP_4F22829C */
