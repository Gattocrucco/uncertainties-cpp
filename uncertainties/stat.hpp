// stat.hpp
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

#ifndef UNCERTAINTIES_STAT_HPP_C71B9D96
#define UNCERTAINTIES_STAT_HPP_C71B9D96

/*! \file
\brief Defines functions to manipulate covariance matrices.
*/

#include <iterator>
#include <stdexcept>
#include <string>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        template<typename InputIt, typename OutputIt, typename Operation>
        OutputIt outer(InputIt begin, InputIt end, OutputIt matrix,
                       Operation op, Order order=Order::row_major) {
            for (InputIt i = begin; i != end; ++i) {
                for (InputIt j = begin; j != end; ++j) {
                    *matrix = order == Order::row_major ? op(*i, *j) : op(*j, *i);
                    ++matrix;
                }
            }
            return matrix;
        }

        template<typename OutVector, typename InVector, typename Operation>
        OutVector outer(const InVector &x, Operation op, Order order=Order::row_major) {
            const typename InVector::size_type n = x.size();
            OutVector matrix(n * n);
            internal::outer(std::begin(x), std::end(x), std::begin(matrix), op, order);
            return matrix;
        }
    }
    
    template<typename InputIt, typename OutputIt>
    OutputIt cov_matrix(InputIt begin, InputIt end, OutputIt matrix,
                        const Order order=Order::row_major) {
        using Type = typename InputIt::value_type;
        return internal::outer(begin, end, matrix, [](const Type &x, const Type &y) {
            return cov(x, y);
        }, order);
    }
    
    template<typename OutVector, typename InVector>
    OutVector cov_matrix(const InVector &x, const Order order=Order::row_major) {
        using Type = typename InVector::value_type;
        return internal::outer<OutVector>(x, [](const Type &x, const Type &y) {
            return cov(x, y);
        }, order);
    }
    
    template<typename InputIt, typename OutputIt>
    OutputIt corr_matrix(InputIt begin, InputIt end, OutputIt matrix,
                         const Order order=Order::row_major) {
        using Type = typename InputIt::value_type;
        return internal::outer(begin, end, matrix, [](const Type &x, const Type &y) {
            return corr(x, y);
        }, order);
    }
    
    template<typename OutVector, typename InVector>
    OutVector corr_matrix(const InVector &x, const Order order=Order::row_major) {
        using Type = typename InVector::value_type;
        return internal::outer<OutVector>(x, [](const Type &x, const Type &y) {
            return corr(x, y);
        }, order);
    }
    
    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector corr2cov(const InVectorA &sigma, const InVectorB &corr,
                       const bool switch_order=false) {
        using Index = typename InVectorB::size_type;
        const Index n = sigma.size();
        if (corr.size() != n * n) {
            throw std::invalid_argument(
                "uncertainties::corr2cov: vector size " + std::to_string(n) +
                " not compatible with matrix buffer size "
                + std::to_string(corr.size())
            );
        }
        OutVector cov(n * n);
        for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j < n; ++j) {
                using Real = typename InVectorB::value_type;
                const Real &c = switch_order ? corr[n * j + i] : corr[n * i + j];
                cov[n * i + j] = sigma[i] * sigma[j] * c;
            }
        }
        return cov;
    }
    
    template<typename OutVector, typename InVector>
    OutVector nom_vector(const InVector &v) {
        OutVector mu;
        for (const auto &x : v) {
            mu.push_back(nom(x));
        }
        return mu;
    }
}

#endif /* end of include guard: UNCERTAINTIES_STAT_HPP_C71B9D96 */
