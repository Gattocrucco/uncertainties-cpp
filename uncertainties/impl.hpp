// impl.hpp
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

#ifndef UNCERTAINTIES_IMPL_HPP_8AC76B25
#define UNCERTAINTIES_IMPL_HPP_8AC76B25

/*! \file
\brief Import this header in one (and only one) source file of your choice.

This header contains non-template code to be compiled once. 
*/

#include <string>
#include <cstdlib>
#include <atomic>
// #include <array>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        std::atomic<Id> last_id {0}; // must be >= 0

        void insert_dot(std::string *s, int n, int e) {
            e += s->size() - n;
            n = s->size();
            if (e >= n - 1) {
                // no dot at end of mantissa
            } else if (e >= 0) {
                s->insert(1 + e, 1, '.');
            } else if (e <= -1) {
                s->insert(0, -e, '0');
                s->insert(1, 1, '.');
            }
        }
        
        std::string format_exp(const int e) {
            return (e > 0 ? "+" : "-") + std::to_string(std::abs(e));
        }
        
        // std::array<std::array<int, 9>, 9> compute_binom_coeffs() {
        //     std::array<std::array<int, 9>, 9> C;
        //     for (int n = 0; n <= 8; ++n) {
        //         for (int k = 0; k <= n; ++k) {
        //             if (n == k or k == 0) {
        //                 C[n][k] = 1;
        //             } else {
        //                 C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
        //             }
        //         }
        //     }
        //     return C;
        // }
    }
}

#endif /* end of include guard: UNCERTAINTIES_IMPL_HPP_8AC76B25 */
