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

#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <utility>
#include <memory>
#include <type_traits>

#include "core.hpp"
#include "internal/hessgrad.hpp"
#include "internal/moments.hpp"

namespace uncertainties {
    namespace internal {
        inline int propsign(const Prop prop) noexcept {
            return prop == Prop::est ? -1 : 1;
        }
    }
    
    template<typename Real, Prop prop>
    class UReal2 {
    private:
        using HessGrad = internal::HessGrad<Real>;
        using Diag = typename HessGrad::Diag;
        using ConstDiagIt = typename HessGrad::ConstDiagIt;
        using ConstTriIt = typename HessGrad::ConstTriIt;
        using ConstTriItNoSkip = typename HessGrad::ConstTriItNoSkip;
        
        // variables
        HessGrad hg;
        Real mu;
        std::array<Real, 3> mom;
        std::array<bool, 3> mom_to_compute;
        
    public:
        using real_type = Real;
        static constexpr Prop prop_mode = prop;
        
        UReal2(const Real &n, const std::array<Real, 7> &moments):
        mu {n} {
            const Id id = ++internal::last_id;
            Diag &diag = this->hg.diag(id);
            diag.grad = 1;
            diag.hhess = 0;
            diag.mom = internal::M7P<Real>(new std::array<Real, 7>(moments));
            std::copy(moments.begin(), moments.begin() + 3, this->mom.begin());
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

        inline const Real &n() const noexcept {
            return this->mu;
        }

        Real s() {
            using std::sqrt;
            return sqrt(this->m(2));
        }

        Real s() const {
            using std::sqrt;
            return sqrt(this->m(2));
        }
        
        Real skew() {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        Real skew() const {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        Real kurt() {
            const Real &s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        Real kurt() const {
            const Real &s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        Real m(const int n) {
            const int i = n - 2;
            if (this->mom_to_compute.at(i)) {
                this->mom[i] = internal::compute_mom(this->hg, n);
                this->mom_to_compute[i] = false;
            }
            return this->mom[i];
        }
        
        Real m(const int n) const {
            const int i = n - 2;
            if (this->mom_to_compute.at(i)) {
                return internal::compute_mom(this->hg, n);
            } else {
                return this->mom[i];
            }
        }

        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        friend UReal2<Real, prop> change_nom(const UReal2<Real, prop> &x, const Real &n) {
            UReal2<Real, prop> y = x;
            y.mu = n;
            return y;
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
        
        friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            return 0;
        }
        
        template<typename Number>
        friend UReal2<Real, prop> unary(Number &x, const Real &fx, const Real &dfdx, const Real &ddfdxdx) {
            static_assert(
                std::is_same<typename std::remove_cv<Number>::type,
                             UReal2<Real, prop>
                             >::value,
                "can not apply on different type"
            );
            UReal2<Real, prop> result;
            const ConstDiagIt dend = x.hg.cdend();
            for (ConstDiagIt it = x.hg.cdbegin(); it != dend; ++it) {
                Diag &dstdiag = result.hg.diag(it->first);
                const Diag &srcdiag = it->second;
                dstdiag.grad = srcdiag.grad * dfdx;
                dstdiag.hhess = (ddfdxdx / 2) * srcdiag.grad * srcdiag.grad;
                dstdiag.hhess += dfdx * srcdiag.hhess;
                dstdiag.mom = srcdiag.mom;
            }
            if (ddfdxdx != 0) {
                result.mu = fx;
                const ConstTriItNoSkip tend = x.hg.ctnend();
                for (ConstTriItNoSkip it = x.hg.ctnbegin(); it != tend; ++it) {
                    Real &dsthhess = result.hg.tri(it.id1(), it.id2());
                    const Real &srchhess = *it;
                    dsthhess = (ddfdxdx / 2) * it.diag1().grad * it.diag2().grad;
                    dsthhess = dfdx * srchhess;
                }
            } else {
                result.mu = fx + internal::propsign(prop) * ddfdxdx * x.m(2) / 2;
                const ConstTriIt tend = x.hg.ctend();
                for (ConstTriIt it = x.hg.ctbegin(); it != tend; ++it) {
                    Real &dsthhess = result.hg.tri(it.id1(), it.id2());
                    const Real &srchhess = *it;
                    dsthhess = dfdx * srchhess;
                }
            }
            for (bool &b : result.mom_to_compute) {
                b = true;
            }
            return result;
        }
        
        inline const UReal2<Real, prop> &operator+() const noexcept {
            return *this;
        }
    };
    
    template<typename Number>
    Number operator-(Number &x) {
        return unary(x, -x.n(), -1, 0);
    }
    
    using udouble2e = UReal2<double, Prop::est>;
    using udouble2m = UReal2<double, Prop::mean>;
    
    using ufloat2e = UReal2<float, Prop::est>;
    using ufloat2m = UReal2<float, Prop::mean>;
    
    template<typename Real, Prop prop>
    const Real &nom(const UReal2<Real, prop> &x) {
        return x.n();
    }

    template<typename Real, Prop prop>
    Real sdev(const UReal2<Real, prop> &x) {
        return x.s();
    }

    template<typename Real, Prop prop>
    Real sdev(UReal2<Real, prop> &x) {
        return x.s();
    }
}

#endif /* end of include guard: UNCERTAINTIES_UREAL2_HPP_4F22829C */
