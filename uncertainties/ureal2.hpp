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

#include "core.hpp"
#include "internal/hessgrad.hpp"
#include "internal/moments.hpp"

namespace uncertainties {
    template<typename Real, Prop prop>
    class UReal2 {
    private:        
        // variables
        internal::HessGrad<Real> hg;
        Real mu;
        std::array<Real, 3> mom;
        std::array<bool, 3> mom_cached;
        
    public:
        using real_type = Real;
        static constexpr Prop prop_mode = prop;
        
        UReal2(const Real &mu, const std::array<Real, 7> &moments) {
            const Id id = ++internal::last_id;
            this->mu = mu;
            using Diag = typename internal::HessGrad<Real>::Diag;
            Diag &diag = this->hg.diag(id);
            diag.grad = 1;
            diag.dhess = 0;
            diag.mom = internal::M7P<Real>(new std::array<Real, 7>(moments));
        }

        UReal2(const Real &mu) {
            this->mu = mu;
        }

        UReal2() {
            ;
        }
        
        inline bool isindep() const noexcept {
            return this->hg.at_most_one_grad();
        }

        Id indepid() const noexcept {
            if (not this->isindep()) {
                return invalid_id;
            } else if (this->hg.size() == 1) {
                return this->hg.dbegin()->first;
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
            if (not this->mom_cached.at(i)) {
                this->mom[i] = internal::compute_mom(this->hg, n);
                this->mom_cached[i] = true;
            }
            return this->mom[i];
        }
        
        Real m(const int n) const {
            const int i = n - 2;
            if (not this->mom_cached.at(i)) {
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
        
        template<typename OtherReal>
        operator UReal2<OtherReal, prop>() const {
            UReal2<OtherReal, prop> x;
            x.hg = this->hg;
            x.mu = this->mu;
            std::copy(this->mom.begin(), this->mom.end(), x.mom.begin());
            x.mom_cached = this->mom_cached;
            return x;
        }
        
        template<Prop other_prop>
        explicit UReal2(const UReal2<Real, other_prop> &x) {
            this->hg = x.hg;
            this->mu = x.mu;
            std::copy(x.mom.begin(), x.mom.end(), this->mom.begin());
            this->mom_cached = x.mom_cached;
        }
        
        friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            return 0;
        }
    };
    
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
