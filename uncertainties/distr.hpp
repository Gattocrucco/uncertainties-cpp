// distr.hpp
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

#ifndef UNCERTAINTIES_DISTR_HPP_FC01FB72
#define UNCERTAINTIES_DISTR_HPP_FC01FB72

/*! \file
\brief Distributions to initialize `UReal2` easily.

The distributions are represented by functions with one template parameter that
must be a specialization of `UReal2`. They just create an `UReal2` with the
first eight moments of the specified distribution. All functions are in
namespace `distr`. Example:

~~~cpp
#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
...
namespace unc = uncertainties;
unc::udouble2e x = unc::distr::normal<unc::udouble2e>(0, 1);
~~~

*/

#include <array>
#include <cassert>
#include <cmath>
#include <sstream>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        inline int binom_coeff(const int n, const int k) {
            assert(0 <= n and n <= 8);
            assert(0 <= k and k <= n);
            static const std::array<std::array<int, 9>, 9> coeffs
                = compute_binom_coeffs();
            return coeffs[n][k];
        }

        template<typename Real>
        std::array<Real, 7> central_moments(const std::array<Real, 8> &zm) {
            std::array<Real, 9> mun {1};
            for (int i = 1; i <= 8; ++i) {
                mun[i] = mun[i - 1] * zm[0];
            }
            std::array<Real, 7> cm;
            for (int n = 2; n <= 8; ++n) {
                const int i = n - 2;
                const int nsign = n % 2 == 0 ? 1 : -1;
                cm[i] = nsign * (1 - n) * mun[n]; // k = 0, 1
                for (int k = 2; k < n; ++k) {
                    const int j = k - 1;
                    const int ksign = k % 2 == 0 ? 1 : -1;
                    cm[i] += nsign * ksign * binom_coeff(n, k) * zm[j] * mun[n - k];
                }
                cm[i] += zm[i + 1]; // k = n
            }
            return cm;
        }
        
        template<typename Real>
        std::array<Real, 6> std_moments(const std::array<Real, 7> &m) {
            std::array<Real, 6> stdm;
            using std::sqrt;
            const Real s = sqrt(m[0]);
            Real sn = m[0];
            for (int i = 0; i < 6; ++i) {
                sn *= s;
                stdm[i] = m[i + 1] / sn;
            }
            return stdm;
        }
    }
    
    /*!
    \brief Namespace for the statistical distributions.
    */
    namespace distr {
        /*!
        \brief Normal (gaussian) with mean `mu` and standard deviation `sigma`.
        
        \throw std::invalid_argument if `sigma < 0`.
        */
        template<typename Number>
        Number normal(const typename Number::real_type &mu,
                      const typename Number::real_type &sigma) {
            static const std::array<typename Number::real_type, 6> std_moments {
                0,
                3,
                0,
                3 * 5,
                0,
                3 * 5 * 7,
            };
            return Number(mu, sigma, std_moments);
        }
        
        /*!
        \brief Chisquare with mean `k`.
        
        \throw std::invalid_argument if `k < 0`.
        */
        template<typename Number>
        Number chisquare(const int k) {
            if (k < 0) {
                std::ostringstream ss;
                ss << "uncertainties::distr::chisquare: k = " << k << " < 0";
                throw std::invalid_argument(ss.str());
            }
            if (k == 0) {
                return Number(0);
            }
            using Real = typename Number::real_type;
            const Real K = k;
            using std::sqrt;
            const Real sqrtK = sqrt(K);
            static const Real sqrt2 = sqrt(Real(2));
            const std::array<Real, 6> std_moments {
                2*sqrt2/sqrtK,
                3 + 12/K,
                sqrt2*(20 + 48/K)/sqrtK,
                15 + (260 + 480/K)/K,
                sqrt2*((2880/K + 1848)/K + 210)/sqrtK,
                105 + ((40320/K + 29232)/K + 4760)/K
            };
            return Number(K, sqrt2 * sqrtK, std_moments);
        }
        
        /*!
        \brief Uniform distribution on the interval [`left`, `right`].
        
        \throw std::invalid_argument if `left > right`.
        */
        template<typename Number>
        Number uniform(const typename Number::real_type &left,
                       const typename Number::real_type &right) {
            using Real = typename Number::real_type;
            static const std::array<Real, 6> std_moments {
                0,
                3 * 3 / Real(5),
                0,
                3 * 3 * 3 / Real(7),
                0,
                3 * 3, // * 3 * 3 / Real(9),
            };
            const Real mu = (right + left) / 2;
            using std::sqrt;
            const Real sigma = sqrt(Real(1) / (3 * 2 * 2)) * (right - left);
            return Number(mu, sigma, std_moments);
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_DISTR_HPP_FC01FB72 */
