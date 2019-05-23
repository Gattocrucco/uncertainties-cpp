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
#include <cassert>
#include <limits>

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
        using ConstHessIt = typename HessGrad::ConstHessIt;
        
        // variables
        HessGrad hg;
        Real mu {0};
        std::array<Real, 4> mom {0, 0, 0, 0};
        std::array<bool, 4> mom_to_compute {false, false, false, false};
        
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
            std::copy(moments.begin(), moments.begin() + 3, this->mom.begin() + 1);
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

        Id indepid() const {
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
        
        Real n() {
            return this->mu + internal::propsign(prop) * this->m(1);
        }

        Real n() const {
            return this->mu + internal::propsign(prop) * this->m(1);
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
            const int i = n - 1;
            if (this->mom_to_compute.at(i)) {
                this->mom[i] = internal::compute_mom(this->hg, n);
                this->mom_to_compute[i] = false;
            }
            return this->mom[i];
        }
        
        Real m(const int n) const {
            const int i = n - 1;
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
        
        Real _grad(const UReal2<Real, prop> &x) const {
            assert(x.hg.size() == 1);
            return this->hg.diag_get(x.hg.cdbegin()->first).grad * x.hg.cdbegin()->second.grad;
        }
        
        Real _hess(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) const {
            assert(x.hg.size() == 1);
            assert(y.hg.size() == 1);
            const Id idx = x.hg.cdbegin()->first;
            const Id idy = y.hg.cdbegin()->first;
            return 2 * (idx == idy ? this->hg.diag_get(idx).hhess : this->hg.tri_get(idx, idy).hhess);
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
        
            ConstHessIt itx = x.hg.chbegin(hddfdxdx == 0 and ddfdxdy == 0);
            const ConstHessIt xend = x.hg.chend();
            ConstHessIt ity = y.hg.chbegin(hddfdydy == 0 and ddfdxdy == 0);
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
                
                if (idx == idy) {
                    *hhess += (ddfdxdy / 2) * 
                        (itx.diag1().grad * ity.diag2().grad + 
                         itx.diag2().grad * ity.diag1().grad);
                }
                
                if (idx <= idy) ++itx;
                if (idy <= idx) ++ity;
            }

            return result;
        }
        
        inline const UReal2<Real, prop> &operator+() const noexcept {
            return *this;
        }
    };
    
    template<typename Real, Prop prop>
    UReal2<Real, prop>
    operator-(const UReal2<Real, prop> &x) {
        return unary(x, -x.first_order_n(), -1, 0);
    }
    
    template<typename Real, Prop prop>
    UReal2<Real, prop>
    operator+(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        return binary(x, y,
                      x.first_order_n() + y.first_order_n(),
                      1, 1, 0, 0, 0);
    }
    
    template<typename Real, Prop prop>
    UReal2<Real, prop>
    operator-(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        return binary(x, y,
                      x.first_order_n() - y.first_order_n(),
                      1, -1, 0, 0, 0);
    }
    
    template<typename Real, Prop prop>
    UReal2<Real, prop>
    operator*(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
        const Real xn = x.first_order_n();
        const Real yn = y.first_order_n();
        return binary(x, y,
                      xn * yn,
                      yn, xn,
                      0, 0, 1);
    }

    template<typename Real, Prop prop>
    UReal2<Real, prop>
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

    using udouble2e = UReal2<double, Prop::est>;
    using udouble2m = UReal2<double, Prop::mean>;
    
    using ufloat2e = UReal2<float, Prop::est>;
    using ufloat2m = UReal2<float, Prop::mean>;
    
    template<typename Real, Prop prop>
    Real nom(const UReal2<Real, prop> &x) {
        return x.n();
    }

    template<typename Real, Prop prop>
    Real nom(UReal2<Real, prop> &x) {
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
