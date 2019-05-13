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
\brief Import this header in a source file of your choice.
*/

#include <string>
#include <cstdlib>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        Id last_id {};

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
    }
}

#endif /* end of include guard: UNCERTAINTIES_IMPL_HPP_8AC76B25 */
