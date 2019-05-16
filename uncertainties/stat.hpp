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

The covariance matrix of a vector of variables \f$ (x_1, \ldots, x_n) \f$ is
defined as \f$ V_{ij} = \mathrm{Cov}(x_i, x_j) \f$. Similarly for the
correlation matrix.

Iterators or vectors
--------------------

The functions defined here come in two flavors: the list of variables can either
be specified with iterators or using a `std::vector`-like class. Example using
iterators:

~~~cpp
#include <vector>
#include <uncertainties/ureal.hpp>
#include <uncertainties/stat.hpp>
...
namespace unc = uncertainties;
std::vector<unc::udouble> x;
// fill x in some way...
std::vector<double> cov(x.size() * x.size());
unc::cov_matrix(x.begin(), x.end(), cov.begin());
// important: we had to set the length of `cov` beforehand
~~~

Example using directly a vector for the output:

~~~cpp
std::vector<double> corr;
corr = unc::corr_matrix<std::vector<double>>(x);
~~~

If the flexibility of iterators is not needed, the second version is
recommended because it is simpler and safer.

Note on matrix storage order
----------------------------

A matrix has two indices but is stored in a linear sequence. There are two
conventional ways to arrange a matrix in an array: row-major and column-major.
Let \f$ M \f$ be an \f$ n \times m \f$ matrix and \f$ A \f$ a \f$ n \cdot
m\f$-long array where \f$ M \f$ has to be stored. Row-major means that \f$
M_{ij} \f$ goes to \f$ A_{m \cdot i + j} \f$ while column-major means it goes
to \f$ A_{i + n \cdot j} \f$.

You see that, when reading back an array as a matrix, using the wrong order
means transposing the matrix (if it is square, otherwise just a mess).
Since covariance and correlation matrices are symmetrical, the storage order
is thus irrelevant.
*/

#include <iterator>
#include <stdexcept>
#include <string>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        template<typename InputIt, typename OutputIt, typename Operation>
        OutputIt outer(InputIt begin, InputIt end, OutputIt matrix,
                       Operation op) {
            for (InputIt i = begin; i != end; ++i) {
                for (InputIt j = begin; j != end; ++j) {
                    *matrix = op(*i, *j);
                    ++matrix;
                }
            }
            return matrix;
        }

        template<typename OutVector, typename InVector, typename Operation>
        OutVector outer(const InVector &x, Operation op) {
            const typename InVector::size_type n = x.size();
            OutVector matrix(n * n);
            internal::outer(std::begin(x), std::end(x), std::begin(matrix), op);
            return matrix;
        }
    }
    
    /*!
    \brief Compute the covariance matrix of the sequence [`begin`, `end`).
    
    The output is written into the sequence starting at `matrix`, which shall
    have the square of the the size of [`begin`, `end`).
    
    `InputIt` shall be an iterator to a type for which `cov` is defined and
    `OutputIt` shall be an iterator to a type to which the output of such
    function `cov` can be copied.
    */
    template<typename InputIt, typename OutputIt>
    OutputIt cov_matrix(InputIt begin, InputIt end, OutputIt matrix) {
        using Type = typename InputIt::value_type;
        return internal::outer(begin, end, matrix, [](const Type &x, const Type &y) {
            return cov(x, y);
        });
    }
    
    /*!
    \brief Compute and return the covariance matrix of the vector `x`.
    
    `InVector` shall be a forward-iterable sequence type with the member
    function `size`. `OutVector` shall be a forward-iterable sequence type that
    has a constructor taking the initial size.
    */
    template<typename OutVector, typename InVector>
    OutVector cov_matrix(const InVector &x) {
        using Type = typename InVector::value_type;
        return internal::outer<OutVector>(x, [](const Type &x, const Type &y) {
            return cov(x, y);
        });
    }
    
    /*!
    \brief Compute the correlation matrix of the sequence [`begin`, `end`).
    
    The output is written into the sequence starting at `matrix`, which shall
    have the square of the the size of [`begin`, `end`).
    
    `InputIt` shall be an iterator to a type for which `corr` is defined and
    `OutputIt` shall be an iterator to a type to which the output of such
    function `corr` can be copied.
    */
    template<typename InputIt, typename OutputIt>
    OutputIt corr_matrix(InputIt begin, InputIt end, OutputIt matrix) {
        using Type = typename InputIt::value_type;
        return internal::outer(begin, end, matrix, [](const Type &x, const Type &y) {
            return corr(x, y);
        });
    }
    
    /*!
    \brief Compute and return the correlation matrix of the vector `x`.
    
    `InVector` shall be a forward-iterable sequence type with the member
    function `size`. `OutVector` shall be a forward-iterable sequence type that
    has a constructor taking the initial size.
    */
    template<typename OutVector, typename InVector>
    OutVector corr_matrix(const InVector &x) {
        using Type = typename InVector::value_type;
        return internal::outer<OutVector>(x, [](const Type &x, const Type &y) {
            return corr(x, y);
        });
    }
    
    /*!
    \brief Convert a correlation matrix to a covariance matrix.
    
    The covariance matrix is obtained with:
    
    \f[
    \mathtt{cov}_{ij}
    = \mathtt{corr}_{ij} \cdot \mathtt{sigma}_i \cdot \mathtt{sigma}_j.
    \f]
    
    `InVectorA` and `InVectorB` shall be forward-iterable sequence types with
    the member function `size`. `OutVector` shall be a sequence type that has a
    constructor taking the initial size and indexing with `operator[]`.
    */
    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector corr2cov(const InVectorA &sigma, const InVectorB &corr) {
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
                cov[n * i + j] = sigma[i] * sigma[j] * corr[n * i + j];
            }
        }
        return cov;
    }
    
    /*!
    \brief Get the means of a vector of variables.
    
    Applies `nom` element-wise to the vector `v`.
    
    `InVector` must a forward-iterable type. `OutVector` must be
    default-costructible and have the method `push_back`.
    */
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
