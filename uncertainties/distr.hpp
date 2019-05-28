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

#include <array>
#include <cassert>

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
    }
    
    namespace distr {
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
        
        template<typename Number>
        Number chisquare(const int k) {
            return Number();
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_DISTR_HPP_FC01FB72 */
