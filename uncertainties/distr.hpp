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

#include "core.hpp"

namespace uncertainties {
    namespace distr {
        template<typename Number>
        Number normal(const typename Number::real_type &mu,
                      const typename Number::real_type &sigma) {
            std::array<typename Number::real_type, 7> moments;
            moments[0] = sigma * sigma;
            moments[2] = 3 * moments[0] * moments[0];
            moments[4] = 5 * moments[2] * moments[0];
            moments[6] = 7 * moments[4] * moments[0];
            return Number(mu, moments);
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_DISTR_HPP_FC01FB72 */
