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
            diag.mom = internal::Moments<Real>(new std::array<Real, 6>);
            if (diag.grad > 0) {
                Real sn = moments[0];
                for (int i = 0; i < 6; ++i) {
                    sn *= diag.grad;
                    (*diag.mom)[i] = moments[i + 1] / sn;
                }
            } else {
                diag.mom->fill(0);
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
            diag.mom = internal::Moments<Real>(new std::array<Real, 6>(std_moments));
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
            const int i = n - 1;
            if (i < 0 or i > 3) {
                std::ostringstream s;
                s << "uncertainties::UReal2::m: ";
                s << "moment order " << n << " out of range";
                throw std::invalid_argument(s.str());
            }
            if (this->mom_to_compute[i]) {
                this->mom[i] = internal::compute_mom(this->hg, n);
                this->mom_to_compute[i] = false;
            }
            return this->mom[i];
        }
        
        Real m(const int n) const {
            const int i = n - 1;
            if (i < 0 or i > 3) {
                std::ostringstream s;
                s << "uncertainties::UReal2::m const: ";
                s << "moment order " << n << " out of range";
                throw std::invalid_argument(s.str());
            }
            if (this->mom_to_compute[i]) {
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
        
        inline friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) noexcept {
            return internal::compute_c2(x.hg, y.hg);
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
}

#endif /* end of include guard: UNCERTAINTIES_UREAL2_HPP_4F22829C */
