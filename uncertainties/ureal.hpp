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
\brief Namespace for all the definitions of the library.
*/
namespace uncertainties {
    /*!
    \brief Represents a number with associated uncertainty.
    
    Basics
    ======
    
    An `UReal` represents a statistical variable of which the first two moments
    of the probability distribution are known: the mean and the variance. When
    doing a computation, the mean and variance of the result can be computed to
    first order using only the mean and variance of the operands. First order
    means approximating the function as linear.
    
    The template parameter `Real`, which is aliased to the member type
    `real_type`, is the numerical type used. Two aliases are provided for
    convenience:
    
    ~~~cpp
    // (in namespace uncertainties)
    using udouble = UReal<double>;
    using ufloat = UReal<float>;
    ~~~
    
    Initialization
    --------------
    
    An `UReal` can be constructed with 0, 1 or 2 numbers with intuitive
    semantics:
    
    ~~~cpp
    namespace unc = uncertainties;
    unc::udouble x; // mean = 0, standard deviation = 0
    unc::udouble x = 1; // mean = 1, standard deviation = 0
    unc::udouble y = {1, 2}; // mean = 1, standard deviation = 2
    ~~~
    
    When doing operations with regular numbers, the non-`UReal` operands are
    treated as having null variance.
    
    Accessing the moments
    ---------------------
    
    The mean and standard deviation (square root of the variance) of an `UReal`
    can be accessed with member functions or external functions, while the
    variance is a friend function:
    
    ~~~cpp
    unc::udouble x;
    double mean = x.n();
    double std = x.s();
    mean = unc::nom(x);
    std = unc::sdev(x);
    double v = var(x); // = std * std
    ~~~
    
    The covariance and correlation can be computed with friend functions:
    
    ~~~cpp
    unc::udouble x(1, 0.3);
    unc::udouble y(2, 0.4);
    double c = cov(x, x); // = var(x) = 0.09
    c = cov(x, y); // 0
    c = cov(x, y + x); // 0.09
    c = cov(x, y - x); // -0.09
    c = corr(x, x + y); // = cov(x,x+y)/(sdev(x)*sdev(x+y)) = 0.6
    ~~~
    
    Printing
    --------
    
    An `UReal` can be formatted to string and output to a stream. You have to
    include the header `io.hpp`.
    
    ~~~cpp
    #include <uncertainties/io.hpp>
    ...
    unc::udouble x = {13, 0.4};
    std::cout << x << "\n"; // will print "13.0 ± 0.4"
    std::cout << x.format(3) << "\n"; // "13.000 ± 0.400"
    std::cout << format(x, 3) << "\n"; // the same
    ~~~
    
    See the function `format` in `io.hpp` for details.
    
    Dependent vs independent variables
    ----------------------------------
    
    When a `UReal` is initialized with mean and standard deviation, it is an
    _independent variable_. Conversely, values obtained through operations on
    `UReal`s are _dependent variables_. Dependent variables store internally
    information on all the independent variables that entered the computation.
    The (in)dependency of a variable can be tested with the member function
    `isindep`:
    
    ~~~cpp
    unc::udouble x = {1, 0.1};
    bool ind = x.isindep(); // true
    unc::udouble y = x + x;
    ind = y.isindep(); // false
    ~~~
    
    A value uniquely identifying an independent variable can be obtained with
    `indepid`:
    
    ~~~cpp
    unc::Id id = x.indepid(); // some value
    id = y.indepid(); // returns unc::invalid_id
    ~~~
    
    Note that a variable that is formally dependent but really independent
    may be generated:
    
    ~~~cpp
    unc::udouble z = x - x; // 0 +/- 0, exactly zero
    id = z.isindep(); // false
    ~~~
    
    Implementing functions
    ======================
    
    The header `math.hpp` defines standard mathematical functions on `UReal`s.
    Other functions can be easily added with the utilities in `functions.hpp`.
    
    If that is not sufficient, you can use directly the friend functions
    `unary`, `binary`, `nary` and the member function `binary_assign`. They
    require you to compute manually the numbers required. The mean of the
    result is just the computation carried on on the means of the operands as
    usual. To propagate the uncertainty, the derivative of the function
    computed at the means is required.
    
    Example re-implementing sin:
    
    ~~~cpp
    #include <cmath>
    ...
    unc::udouble usin(unc::udouble &x) {
        double mean = sin(x.n());
        double dsindx = cos(x.n());
        return unary(x, mean, dsindx);
    }
    ~~~
    
    Example re-implementing multiplication:
    
    ~~~cpp
    unc::udouble umult(unc::udouble &x, unc::udouble &y) {
        double mean = x.n() * y.n();
        double dmdx = y.n();
        double dmdy = x.n();
        return binary(x, y, mean, dmdx, dmdy);
    }
    ~~~
    
    Internals
    =========
    
    Independent id
    --------------
    
    Independent ids are integers. Each time a non-zero uncertainty `UReal` is
    initialized, a global counter is increased. The counter is thread-safe.
    
    Dependency tracking
    -------------------
    
    
    
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
