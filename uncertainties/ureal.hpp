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
\brief Defines class template `UReal`.
*/

#include <map>
#include <string>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <cassert>
#include <limits>
#include <sstream>

#include "core.hpp"

/*!
\brief Namespace for all the definitions of the library.
*/
namespace uncertainties {
    /*!
    \brief Class for first order error propagation.
    
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
    unc::udouble y = {0.5, 0.2};
    unc::udouble z = x + y;
    ind = z.isindep(); // false
    ~~~
    
    A value uniquely identifying an independent variable can be obtained with
    `indepid`:
    
    ~~~cpp
    unc::Id id = x.indepid(); // some value
    id = y.indepid(); // a value different from the one of x
    id = z.indepid(); // returns unc::invalid_id
    ~~~
    
    If a computation involves only one variable, the result still classifies as
    independent, in particular a copy of an independent variable is still
    independent although it is completely correlated with the original:
    
    ~~~cpp
    z = x + x;
    id = z.isindep(); // true
    y = x;
    x.isindep(); // true
    y.isindep(); // true
    corr(x, y); // 1.0
    ~~~
        
    So the property of an independent variable is that it has either zero
    or full (1 or -1) correlation with any other independent variable.
    
    Efficiency
    ==========
    
    A dependent variable stores its dependency to all the independent variables
    that entered the computations. If a variable depends on \f$ N \f$
    independent variables, operations on the variable in general are \f$ O(N)
    \f$. Example of a simple but *quadratical* computation:
    
    ~~~cpp
    unc::udouble x = 0.0;
    for (int i = 0; i < 1000; ++i)
        x = x + unc::udouble(1.0, 0.1);
    ~~~
    
    This seemingly innocent loop at each iteration creates a temporary variable
    containing the result of `x + {1.0, 0.1}` and copies it back to `x`. At
    iteration `i`, there are `i` independent variable coefficients stored into
    `x`, so the copy takes time `i`. The total number of coefficients copied is
    then about \f$ 1000 \cdot 1000 / 2 = 500000 \f$. The right way to do the
    summation is using the add-assign operator and do `x += unc::udouble(1.0,
    0.1)` in the loop.
    
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
    
    If the uncertainty is zero there is no need to create a new id, because the
    variable will never propagate any uncertainty in calculations.
    
    Dependency tracking
    -------------------
    
    Let \f$ x_i \f$ be independent variables with means \f$ \mu_i \f$ and
    standard deviations \f$ \sigma_i \f$. Then the covariance of \f$ y =
    f(x_1,\ldots,x_n) \f$ and \f$ z = g(x_1,\ldots,x_n) \f$ is, to first order:
    
    \f[
    \mathrm{Cov}(y, z) = \sum_{ij}
    \left(\left.\frac{\partial f}{\partial x_i}\right|_{\mu} \sigma_i\right)
    \left(\left.\frac{\partial g}{\partial x_j}\right|_{\mu} \sigma_j\right).
    \f]
    
    Moreover, let \f$ a_k \f$ be dependent variables, and let \f$ b = h(a_1,
    \ldots, a_m) \f$. Then the derivative of \f$ b \f$ respect to an independent
    variable is obtained by:
    
    \f[
    \left.\frac{\partial b}{\partial x_i}\right|_{\mu}
    = \sum_k \left.\frac{\partial b}{\partial a_k}\right|_{\mu}
    \left.\frac{\partial a_k}{\partial x_i}\right|_{\mu},
    \f]
    
    and so:
    
    \f[
    \left(\left.\frac{\partial b}{\partial x_i}\right|_{\mu}\sigma_i\right)
    = \sum_k \left.\frac{\partial b}{\partial a_k}\right|_{\mu}
    \left(\left.\frac{\partial a_k}{\partial x_i}\right|_{\mu}\sigma_i\right).
    \f]
    
    This implies that we just need to store into a dependent variable \f$ y \f$
    the coefficients \f$ \partial y/\partial x_i \sigma_i \f$ for all the
    independent variables \f$ x_i \f$ it depends on. `UReal<Real>` uses a
    `std::map<Id, Real>` to save these coefficients.
    
    Independent variable optimization
    ---------------------------------
    
    An independent variable with id `i` and standard deviation `s` can be
    implemented by setting its internal map to `{{i, s}}`, but this means that
    independent variables will use the heap. Instead, an `UReal` contains
    explicit variables for the id and the standard deviation. An independent
    `UReal` uses these two variables and leaves the map empty, while a
    dependent `UReal` sets the id to `invalid_id` and uses the standard
    deviation variable as a cache of the standard deviation computed from the
    map.
    
    */
    template<typename Real>
    class UReal {
    private:
        // private variables (there are no other variables around):
        Id id {0};
        Real sdev {0};
        Real mu {0};
        std::map<Id, Real> sigma;
        
        // Summary of class invariants:
        // id == 0 -> sdev == 0, sigma.size() == 0
        // id > 0 -> sigma.size() == 0
        // id < 0, sdev < 0 -> sigma.size() >= 0, sdev completely ignored
        // id < 0, sdev >= 0 -> sigma.size() >= 0, sdev * sdev == s2()
        
        /* Longer explanation:
        
        `id` is the independent variable id (a signed integer). It is set to
        zero if the variable is initialized with a zero standard deviation.
        In this way a variable with `id == 0` can be completely ignored when
        propagating uncertainties.
        
        For new variables with nonzero standard deviation, `id` is set
        to a positive number that is incremented each time. The id of variables
        which are the result of a computation is set to `invalid_id` (a
        negative value).
        
        If id > 0, i.e. the variable is independent, the variable `sdev` does
        not store the standard deviation but the coefficient dy/dx * sigma_x,
        so the standard deviation is the absolute value of `sdev`.
        
        If id < 0, i.e. the variable is dependent, the map `sigma` is used to
        store the pairs {id_x, dy/dx * sigma_x} for all the independent
        variables `x` that entered the computations. The variable `sdev` is used
        as a cache for the standard deviation, if sdev < 0 it has to be computed
        using the private member function `s2`.
        */
        
        Real s2() const {
            // computes the variance
            Real s2(0);
            for (const auto &it : this->sigma) {
                const Real &s = it.second;
                s2 += s * s;
            }
            return s2;
        }
        
    public:
        /*!
        \brief The numerical type used.
        */
        using real_type = Real;
        
        /*!
        \brief Construct a new independent variable with mean `n` and standard
        deviation `s`.
        
        Using this constructor is the only way to generate a new independent
        variable id.
        
        \throws std::invalid_argument if `s < 0`. 
        */
        UReal(const Real &n, const Real &s):
        mu {n}, sdev {s}, id {++internal::last_id} {
            if (this->sdev < 0) {
                throw std::invalid_argument("uncertainties::UReal::UReal: s < 0");
            }
        }
        
        /*!
        \brief Construct a variable with mean `n` and zero variance.
        
        A new independent variable id is *not* generated.
        */
        UReal(const Real &n): mu {n} {
            ;
        }
        
        /*!
        \brief Construct a variable with zero mean and variance.
        
        A new independent variable id is *not* generated.
        */
        UReal() {
            ;
        }
        
        /*!
        \brief Return true if the variable is independent.
        
        Independent implies that the variable can only have 0, +1 or -1
        correlation with any other independent variable. A variable
        is independent just after construction and a computation that uses
        only one variable produces an independent variable.
        */
        inline bool isindep() const noexcept {
            assert(this->id < 0 or this->sigma.size() == 0);
            return this->sigma.size() <= 1;
        }
        
        /*!
        \brief Return the independent variable id if the variable is
        independent.
        
        If it is not, it returns `invalid_id`.
        */
        Id indepid() const noexcept {
            assert(this->id < 0 or this->sigma.size() == 0);
            if (this->id >= 0) {
                return this->id;
            } else if (this->sigma.size() == 0) {
                return 0;
            } else if (this->sigma.size() == 1) {
                return this->sigma.begin()->first;
            } else {
                return invalid_id;
            }
        }
        
        /*!
        \brief Return the mean.
        */
        inline const Real &n() const noexcept {
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
                
        /*!
        \brief Return the standard deviation.
        
        For an independent variable it means just returning a number. If
        the variable is dependent a potentially expensive calculation is
        performed. In this case the result is cached. Note that, if the variable
        is const, the const-qualified version of this function is called and
        it can not cache the result.
        */
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

        /*!
        \brief Format the variable.
        
        This function just calls `format(x, args...)`. The `format` function
        is defined in `io.hpp`. Using this function will produce errors if
        `io.hpp` has not been included.
        */
        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        /*!
        \brief Make a copy of `x` with different mean, keeping correlations.
        
        The result is a variable which is identical to `x` apart from the mean
        which is set to `n`.
        */
        friend UReal<Real> change_nom(const UReal<Real> &x, const Real &n) {
            UReal<Real> y = x;
            y.mu = n;
            return y;
        }
        
        // friendship needed for conversion function below
        template<typename OtherReal>
        friend class UReal;
        
        /*!
        \brief Conversion operator to `UReal`s with different numerical type.
        
        Converting to an `UReal` with different numerical type preserves all
        the correlations.
        */
        template<typename OtherReal>
        operator UReal<OtherReal>() const {
            UReal<OtherReal> x;
            x.id = this->id;
            x.sdev = this->sdev;
            x.mu = this->mu;
            for (const auto &it : this->sigma) {
                x.sigma[it.first] = it.second;
            }
            return x;
        }
        
        /*!
        \brief Compute the covariance between `x` and `y`.
        */
        friend Real cov(const UReal<Real> &x, const UReal<Real> &y) {
            using MapIt = typename std::map<Id, Real>::const_iterator;
            Real cov = 0;
            if (x.id > 0 and y.id > 0) {
                if (x.id == y.id) {
                    cov += x.sdev * y.sdev;
                }
            } else if (x.id < 0 and y.id > 0) {
                const MapIt xit = x.sigma.find(y.id);
                if (xit != x.sigma.end()) {
                    cov += xit->second * y.sdev;
                }
            } else if (x.id > 0 and y.id < 0) {
                const MapIt yit = y.sigma.find(x.id);
                if (yit != y.sigma.end()) {
                    cov += x.sdev * yit->second;
                }
            } else if (x.id < 0 and y.id < 0) {
                // This algorithm has a bad worst case: it should be
                // O(min(Nx, Ny)) but imagine that Nx == 1 and that its id is
                // past all the ids in y. Then all y is iterated before
                // concluding they'are uncorrelated! If I iterate over the
                // shorter one and search the longer one, it becomes
                // O(min(Nx, Ny) log max(Nx, Ny)). I can do better if instead
                // of simply incrementing of one step to reach the other
                // iterator, I could search for it starting from where I am.
                // Look in the reference for std::map, maybe there is
                // something. I could even go beyond C++11 if needed.
                MapIt xit = x.sigma.begin();
                MapIt yit = y.sigma.begin();
                while (xit != x.sigma.end() and yit != y.sigma.end()) {
                    const Id xid = xit->first;
                    const Id yid = yit->first;
                    if (xid == yid) cov += xit->second * yit->second;
                    if (xid <= yid) ++xit;
                    if (yid <= xid) ++yit;
                }
            }
            return cov;
        }
        
        friend Real var(const UReal<Real> &x) {
            if (x.id >= 0 or x.sdev >= 0) {
                return x.sdev * x.sdev;
            } else {
                return x.s2();
            }
        }
        
        /*!
        \brief Compute the variance of `x`.
        
        Note that `cov(x, x) == var(x)`, but `var` is faster. Also
        `var(x) == x.s() * x.s()`. The same caching considerations of the
        member function `s` apply here.
        */
        friend Real var(UReal<Real> &x) {
            if (x.id >= 0) {
                return x.sdev * x.sdev;
            } else if (x.sdev < 0) {
                using std::sqrt;
                x.sdev = sqrt(x.s2());
            }
            return x.sdev * x.sdev;
        }
        
        /*!
        \brief Compute the correlation between `x` and `y`.
        
        The correlation is defined as
        `corr(x, y) == cov(x, y) / (x.s() * y.s())`.
        */
        friend Real corr(const UReal<Real> &x, const UReal<Real> &y) {
            return cov(x, y) / (x.s() * y.s());
        }
        
        /*!
        \brief Compute derivative with respect to `x`.
        
        \throw std::invalid_argument if `x` is not independent or if it has
        no uncertainty information.
        */
        Real grad(const UReal<Real> &x) {
            if (not (x.id > 0 or x.sigma.size() == 1)) {
                std::ostringstream ss;
                ss << "uncertainties::UReal::grad: ";
                ss << "variable is not independent and nontrivial";
                throw std::invalid_argument(ss.str());
            }
            const Id xid = x.id > 0 ? x.id : x.sigma.begin()->first;
            const typename std::map<Id, Real>::const_iterator it = this->sigma.find(xid);
            if (it != this->sigma.end()) {
                const Real &xg = x.id > 0 ? x.sdev : x.sigma.begin()->second;
                return it->second / xg;
            } else {
                return 0;
            }
        }
        
        /*!
        \brief Apply a function of one argument to `x`.
        
        This is a low-level interface. Higher level interfaces are provided
        in the header `functions.hpp`.
        
        The argument `mu` is the mean of the result and should be obtained
        by applying the function to the mean of `x`. The argument `dx` is the
        derivative of the function computed at the mean of `x`.
        */
        friend UReal<Real> unary(const UReal<Real> &x, const Real mu, const Real &dx) {
            UReal<Real> y;
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
        
        /*!
        \brief Apply a function of two arguments to `x` and `y`.
        
        This is a low-level interface. Higher level interfaces are provided
        in the header `functions.hpp`.
        
        The argument `mu` is the mean of the result and should be obtained by
        applying the function to the means of `x` and `y`. The argument `dx` is
        the derivative of the function respect to `x` computed at the means of
        `x` and `y`; similarly for `dy`.
        */
        friend UReal<Real> binary(const UReal<Real> &x, const UReal<Real> &y,
                           const Real &mu,
                           const Real &dx, const Real &dy) {
            UReal<Real> z;
            z.id = invalid_id;
            z.mu = mu;
            if (x.id > 0) {
                z.sigma[x.id] = dx * x.sdev;
            } else if (x.id < 0 and y.id >= 0) {
                for (const auto &it : x.sigma) {
                    z.sigma[it.first] = dx * it.second;
                }
            }
            if (y.id > 0) {
                z.sigma[y.id] += dy * y.sdev;
            } else if (y.id < 0 and x.id >= 0) {
                for (const auto &it : y.sigma) {
                    z.sigma[it.first] += dy * it.second;
                }
            }
            if (x.id < 0 and y.id < 0) {
                typename std::map<Id, Real>::const_iterator xit = x.sigma.begin();
                typename std::map<Id, Real>::const_iterator yit = y.sigma.begin();
                constexpr Id maxid = std::numeric_limits<Id>::max();
                while (xit != x.sigma.end() or yit != y.sigma.end()) {
                    const Id xid = xit != x.sigma.end() ? xit->first : maxid;
                    const Id yid = yit != y.sigma.end() ? yit->first : maxid;
                    Real *p;
                    if (xid <= yid) {
                        p = &z.sigma[xid];
                        *p = dx * xit->second;
                        ++xit;
                    }
                    if (yid <= xid) {
                        if (yid != xid) {
                            p = &z.sigma[yid];
                            *p = 0;
                        }
                        *p += dy * yit->second;
                        ++yit;
                    }
                }
            }
            z.sdev = -1;
            return z;
        }
        
        /*!
        \brief Apply a function of any number of arguments to the sequence
        [`xbegin`, `xend`).
        
        This is a low-level interface, but currently no higher-level interface
        for nary operations is provided in `functions.hpp`.
        
        The argument `mu` is the mean of the result and should be obtained by
        applying the function to the means of [`xbegin`, `xend`). The sequence
        starting at `dxbegin` are the derivatives respect to the variables in
        [`xbegin`, `xend`) computed at their means.
        */
        template<typename XIt, typename DxIt>
        friend UReal<Real> nary(XIt xbegin, XIt xend, const Real mu, DxIt dxbegin) {
            UReal<Real> z;
            z.id = invalid_id;
            z.mu = std::move(mu);
            for (; xbegin != xend; ++xbegin, ++dxbegin) {
                const UReal<Real> &x = *xbegin;
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
                
        /*!
        \brief Apply a function of two arguments to the variable and `x`,
        storing the result in the variable.
        
        The argument `mu` is the mean of the result and should be obtained by
        applying the function to the means of the variable and `x`. The
        argument `dt` is the derivative of the function respect to the variable
        computed at the means of the variable and `x`; similarly for `dx`.
        
        For dependent variables this can be significantly more efficient than
        using `binary` and then assigning the result to the first operand.
        */
        const UReal<Real> &binary_assign(const UReal<Real> &x, const Real &mu,
                                         const Real &dt, const Real &dx) {
            if (&x == this) {
                const Real d = dt + dx;
                for (auto &it : this->sigma) {
                    it.second *= d;
                }
                if (this->sdev > 0 or this->id > 0) {
                    this->sdev *= d;
                }
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
            this->mu = mu; // keep this last in case &x == this
            return *this;
        }

        friend inline const UReal<Real> &operator+(const UReal<Real> &x) noexcept {
            return x;
        }
        friend UReal<Real> operator-(const UReal<Real> &x) {
            return unary(x, -x.mu, -1);
        }
        friend UReal<Real> operator+(const UReal<Real> &x, const UReal<Real> &y) {
            return binary(x, y, x.mu + y.mu, 1, 1);
        }
        friend UReal<Real> operator-(const UReal<Real> &x, const UReal<Real> &y) {
            return binary(x, y, x.mu - y.mu, 1, -1);
        }
        friend UReal<Real> operator*(const UReal<Real> &x, const UReal<Real> &y) {
            return binary(x, y, x.mu * y.mu, y.mu, x.mu);
        }
        friend UReal<Real> operator/(const UReal<Real> &x, const UReal<Real> &y) {
            const Real inv_y = Real(1) / y.mu;
            const Real mu = x.mu * inv_y;
            return binary(x, y, mu, inv_y, -mu * inv_y);
        }
        const UReal<Real> &operator+=(const UReal<Real> &x) {
            return binary_assign(x, this->mu + x.mu, 1, 1);
        }
        const UReal<Real> &operator-=(const UReal<Real> &x) {
            return binary_assign(x, this->mu - x.mu, 1, -1);
        }
        const UReal<Real> &operator*=(const UReal<Real> &x) {
            return binary_assign(x, this->mu * x.mu, x.mu, this->mu);
        }
        const UReal<Real> &operator/=(const UReal<Real> &x) {
            const Real inv_x = Real(1) / x.mu;
            const Real mu = this->mu * inv_x;
            return binary_assign(x, mu, inv_x, -mu * inv_x);
        }
    };
    
    /*!
    \brief Return the mean of a variable.
    
    Equivalent to `x.n()`.
    */
    template<typename Real>
    inline const Real &nom(const UReal<Real> &x) noexcept {
        return x.n();
    }
    
    /*!
    \brief Return the standard deviation of a variable.
    
    Equivalent to `x.s()`.
    */
    template<typename Real>
    inline Real sdev(const UReal<Real> &x) {
        return x.s();
    }
        
    using udouble = UReal<double>;
    using ufloat = UReal<float>;

#ifdef UNCERTAINTIES_EXTERN_UDOUBLE
    extern template class UReal<double>;
#endif
}

#endif /* end of include guard: UNCERTAINTIES_UREAL_HPP_07A47EC2 */
