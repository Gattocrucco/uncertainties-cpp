// ureal.hpp
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

#ifndef UNCERTAINTIES_UREAL_HPP_07A47EC2
#define UNCERTAINTIES_UREAL_HPP_07A47EC2

/*! \file
\brief Defines class template `UReal` and basic utilities.
*/

#include <map>
#include <string>
#include <stdexcept>
#include <cmath>
#include <utility>

#include "core.hpp"

/*!
\brief C++ header library for first-order uncertainty propagation.

Basic example:
~~~{.cpp}
#include <iostream>
#include <uncertainties/ureal.hpp>
#include <uncertainties/io.hpp>
#include <uncertainties/impl.hpp>
namespace unc = uncertainties;
int main() {
    unc::udouble x(2, 1), y(2, 1);
    unc::udouble a = x - x;
    unc::udouble b = x - y;
    std::cout << a << ", " << b << "\n";
}
~~~
*/
namespace uncertainties {
    /*!
    \brief Represents a number with associated uncertainty.
    */
    template<typename Real>
    class UReal {
    private:
        using Type = UReal<Real>;
        
        // id == 0 -> sdev == 0, sigma.size() == 0
        // id > 0 -> sigma.size() == 0
        // id < 0, sdev < 0 -> sigma.size() >= 0, sdev completely ignored
        // id < 0, sdev >= 0 -> sigma.size() >= 0, sdev * sdev == s2()
        Id id;
        Real sdev;
        Real mu;
        std::map<Id, Real> sigma;
        
        Real s2() const {
            Real s2(0);
            for (const auto &it : this->sigma) {
                const Real &s = it.second;
                s2 += s * s;
            }
            return s2;
        }
        
    public:
        using real_type = Real;
        
        UReal(const Real n, const Real s):
        mu {std::move(n)}, sdev {std::move(s)}, id {++internal::last_id} {
            if (this->sdev < 0) {
                throw std::invalid_argument("uncertainties::UReal::UReal: s < 0");
            }
        }
        
        UReal(const Real n): mu {std::move(n)} {
            ;
        }
        
        UReal() {
            ;
        }
        
        bool isindep() const noexcept {
            return this->id >= 0;
        }
        
        Id indepid() const noexcept {
            return this->id;
        }
        
        const Real &n() const noexcept {
            return this->mu;
        }
        
        Real s() const {
            if (this->id >= 0) {
                using std::abs;
                return abs(this->sdev);
            } else if (this->sdev >= 0) {
                return this->sdev;
            } else {
                using std::sqrt;
                return sqrt(this->s2());
            }
        }
                
        Real s() {
            if (this->id >= 0) {
                using std::abs;
                return abs(this->sdev);
            } else if (this->sdev < 0) {
                using std::sqrt;
                this->sdev = sqrt(this->s2());
            }
            return this->sdev;
        }

        operator std::string() {
            using std::to_string;
            return to_string(n()) + "+/-" + to_string(s());
        }
        
        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        friend Type copy_unc(const Real n, const Type &x) {
            Type y = x;
            y.mu = std::move(n);
            return y;
        }
        
        template<typename OtherReal>
        friend class UReal;
        
        template<typename OtherReal>
        operator UReal<OtherReal>() {
            UReal<OtherReal> x;
            x.id = this->id;
            x.sdev = this->sdev;
            x.mu = this->mu;
            for (const auto &it : this->sigma) {
                x.sigma[it.first] = it.second;
            }
            return x;
        }
        
        friend Real cov(const Type &x, const Type &y) {
            Real cov(0);
            if (x.id > 0 && y.id > 0) {
                if (x.id == y.id) {
                    cov += x.sdev * y.sdev;
                }
            } else if (x.id < 0 or y.id < 0) {
                const Type *min_size, *max_size;
                if (x.sigma.size() > y.sigma.size()) {
                    min_size = &y;
                    max_size = &x;
                } else {
                    min_size = &x;
                    max_size = &y;
                }
                const auto END = max_size->sigma.end();
                if (min_size->id > 0) {
                    const auto IT = max_size->sigma.find(min_size->id);
                    if (IT != END) {
                        cov += min_size->sdev * IT->second;
                    }
                } else if (min_size->id < 0) {
                    for (const auto &it : min_size->sigma) {
                        if (max_size->id > 0) {
                            if (it.first == max_size->id) {
                                cov += it.second * max_size->sdev;
                            }
                        } else if (max_size->id < 0) {
                            const auto IT = max_size->sigma.find(it.first);
                            if (IT != END) {
                                cov += it.second * IT->second;
                            }
                        }
                    }
                }
            }
            return cov;
        }
        
        friend Real var(const Type &x) {
            if (x.id >= 0 or x.sdev >= 0) {
                return x.sdev * x.sdev;
            } else {
                return x.s2();
            }
        }
        
        friend Real var(Type &x) {
            if (x.id >= 0) {
                return x.sdev * x.sdev;
            } else if (x.sdev < 0) {
                using std::sqrt;
                x.sdev = sqrt(x.s2());
            }
            return x.sdev * x.sdev;
        }

        friend Real corr(const Type &x, const Type &y) {
            return cov(x, y) / (x.s() * y.s());
        }
        
        friend Type unary(const Type &x, const Real mu, const Real &dx) {
            Type y;
            y.id = x.id;
            y.mu = std::move(mu);
            if (x.id > 0) {
                y.sdev = x.sdev * dx;
            } else if (x.id < 0) {
                y.sigma = x.sigma;
                for (auto &it : y.sigma) {
                    it.second *= dx;
                }
                y.sdev = -1;
            }
            return y;
        }
    
        friend Type binary(const Type &x, const Type &y,
                           const Real mu,
                           const Real &dx, const Real &dy) {
            Type z;
            z.id = invalid_id;
            z.mu = std::move(mu);
            if (x.id > 0) {
                z.sigma[x.id] = dx * x.sdev;
            } else if (x.id < 0) {
                for (const auto &it : x.sigma) {
                    z.sigma[it.first] = dx * it.second;
                }
            }
            if (y.id > 0) {
                z.sigma[y.id] += dy * y.sdev;
            } else if (y.id < 0) {
                for (const auto &it : y.sigma) {
                    z.sigma[it.first] += dy * it.second;
                }
            }
            z.sdev = -1;
            return z;
        }
        
        template<typename XIt, typename DxIt>
        friend Type nary(XIt xbegin, XIt xend, const Real mu, DxIt dxbegin) {
            Type z;
            z.id = invalid_id;
            z.mu = std::move(mu);
            for (; xbegin != xend; ++xbegin, ++dxbegin) {
                const Type &x = *xbegin;
                const Real &dx = *dxbegin;
                if (x.id > 0) {
                    z.sigma[x.id] += dx * x.sdev;
                } else if (x.id < 0) {
                    for (const auto &it : x.sigma) {
                        z.sigma[it.first] += dx * it.second;
                    }
                }
            }
            z.sdev = -1;
            return z;
        }
                
        const Type &binary_assign(const Type &x, const Real mu,
                                  const Real &dt, const Real &dx) {
            if (&x == this) {
                const Real d = dt + dx;
                for (auto &it : this->sigma) {
                    it.second *= d;
                }
                this->sdev *= d;
            } else {
                if (this->id > 0) {
                    this->sigma[this->id] = this->sdev;
                }
                this->id = invalid_id;
                if (dt != 1) {
                    for (auto &it : this->sigma) {
                        it.second *= dt;
                    }
                }
                if (x.id > 0) {
                    this->sigma[x.id] += dx * x.sdev;
                } else if (x.id < 0) {
                    for (const auto &it : x.sigma) {
                        this->sigma[it.first] += dx * it.second;
                    }
                }
                this->sdev = -1;
            }
            this->mu = std::move(mu); // keep this last in case &x == this
            return *this;
        }

        friend inline const Type &operator+(const Type &x) noexcept {
            return x;
        }
        friend Type operator-(const Type &x) {
            return unary(x, -x.mu, -1);
        }
        friend Type operator+(const Type &x, const Type &y) {
            return binary(x, y, x.mu + y.mu, 1, 1);
        }
        friend Type operator-(const Type &x, const Type &y) {
            return binary(x, y, x.mu - y.mu, 1, -1);
        }
        friend Type operator*(const Type &x, const Type &y) {
            return binary(x, y, x.mu * y.mu, y.mu, x.mu);
        }
        friend Type operator/(const Type &x, const Type &y) {
            const Real inv_y = Real(1) / y.mu;
            const Real mu = x.mu * inv_y;
            return binary(x, y, mu, inv_y, -mu * inv_y);
        }
        const Type &operator+=(const Type &x) {
            return binary_assign(x, this->mu + x.mu, 1, 1);
        }
        const Type &operator-=(const Type &x) {
            return binary_assign(x, this->mu - x.mu, 1, -1);
        }
        const Type &operator*=(const Type &x) {
            return binary_assign(x, this->mu * x.mu, x.mu, this->mu);
        }
        const Type &operator/=(const Type &x) {
            const Real inv_x = Real(1) / x.mu;
            const Real mu = this->mu * inv_x;
            return binary_assign(x, mu, inv_x, -mu * inv_x);
        }
        
        // template<typename OutVector, typename InVectorA, typename InVectorB>
        // friend OutVector
        // ureals(const InVectorA &, const InVectorB &, const Order);
    };
    
    template<typename Real>
    inline const Real &nom(const UReal<Real> &x) noexcept {
        return x.n();
    }
    
    template<typename Real>
    inline Real sdev(const UReal<Real> &x) {
        return x.s();
    }
        
    using udouble = UReal<double>;
    using ufloat = UReal<float>;
}

#endif /* end of include guard: UNCERTAINTIES_UREAL_HPP_07A47EC2 */
