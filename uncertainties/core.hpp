// core.hpp
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

#ifndef UNCERTAINTIES_CORE_HPP_D4C14D73
#define UNCERTAINTIES_CORE_HPP_D4C14D73

/*! \file
\brief Basic declarations and definitions.
*/

#include <string>
#include <atomic>

namespace uncertainties {
    /*!
    \brief Type of independent variables ids.
    */
    using Id = int; // must be signed
    
    /*!
    \brief Constant returned by `UReal::indepid` if the variable is dependent.
    */
    constexpr Id invalid_id = -1; // must be < 0
    
    namespace internal {
        extern std::atomic<Id> last_id; // defined in impl.hpp
    }
    
    template<typename Real>
    class UReal;
    
    /*!
    \brief Matrix storage order.
    */
    enum class Order {
        row_major,
        col_major
    };
    
    /*!
    \brief For a normal number, `nom` just returns the number.
    */
    template<typename Number>
    inline const Number &nom(const Number &x) noexcept {
        return x;
    }
    
    /*!
    \brief For a normal number, `sdev` returns 0.
    */
    template<typename Number>
    inline Number sdev(const Number &x) noexcept {
        return 0;
    }
    
    /*!
    \brief Convert an `UReal` to a human-readable string.
    */
    template<typename Number>
    std::string format(const Number &x,
                       const float errdig=1.5f,
                       const std::string &sep=" ± ");

    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector ureals(const InVectorA &mu,
                     const InVectorB &cov,
                     const Order order=Order::row_major);
}

#endif /* end of include guard: UNCERTAINTIES_CORE_HPP_D4C14D73 */
