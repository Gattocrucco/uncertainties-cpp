// ureals.hpp
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

#ifndef UNCERTAINTIES_UREALS_HPP_61EB1909
#define UNCERTAINTIES_UREALS_HPP_61EB1909

/*! \file
\brief Defines the function `ureals` to generate a list of correlated variables.

The [Eigen](http://eigen.tuxfamily.org) header library is required to use
this header.
*/

#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <string>
#include <sstream>
#include <cassert>

#include <Eigen/Dense>

#include "core.hpp"

namespace uncertainties {
    /*!
    \brief Generates a sequence of correlated variables given a covariance
    matrix.
    
    `mu` is the vector of means, `cov` is the covariance matrix in row-major
    or column-major ordering. Returns a vector of type `OutVector` containing
    the `UReal`s.
    
    You will have to specify explicitly `OutVector` when using the function:
    
    ~~~cpp
    namespace unc = uncertainties;
    // define `mu` and `cov`...
    std::vector<unc::udouble> x = unc::ureals<std::vector<unc::udouble>>(mu, cov);
    ~~~
    
    \throws std::invalid_argument if the sizes of `mu` and `cov` do not match.
    
    `cov` is not checked to be symmetric. If `cov` is column-major, only the
    lower triangle is read; if it is row-major, the upper triangle.
    
    `OutVector` shall be a default-constructible sequence type with the member
    function `push_back` and the member type `value_type`.
    `OutVector::value_type` shall be an `UReal`-like type. `InVectorA` and
    `InVectorB` shall be sequence types with member functions `size` and
    `operator[]`.
    
    */
    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector ureals(const InVectorA &mu, const InVectorB &cov) {
        // TODO split this function internals in one function diagonalizing the
        // matrix and one building the variables to allow for specialization
        // respect to InVectorB and OutVector
        const std::size_t n = mu.size();
        if (n != 0 ? cov.size() % n != 0 or cov.size() / n != n : cov.size() != 0) {
            throw std::invalid_argument(
                "uncertainties::ureals: vector size " + std::to_string(n) +
                " not compatible with matrix buffer size "
                + std::to_string(cov.size())
            );
        }
        if (n == 0) {
            return OutVector{};
        }
        using UType = typename OutVector::value_type;
        using Real = typename UType::real_type;
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
        Matrix V(n, n);
        // Eigen::Map can use directly cov.data()
        // but then non-vector sequences would not be supported
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = j; i < n; ++i) { // only lower triangle
                V(i, j) = cov[i + n * j];
            }
        }
        const Eigen::LDLT<Matrix> ldlt(V);
        if (ldlt.info() != Eigen::Success) {
            std::ostringstream ss;
            ss << "uncertainties::ureals: ";
            ss << "covariance matrix decomposition failed";
            throw std::runtime_error(ss.str());
        }
        if (not ldlt.isPositive()) {
            std::ostringstream ss;
            ss << "uncertainties::ureals: ";
            ss << "convariance matrix is not positive semidefinite";
            throw std::runtime_error(ss.str());
        }
        const auto L = ldlt.matrixL();
        const auto D = ldlt.vectorD();
        OutVector x;
        for (std::size_t i = 0; i < n; ++i) {
            const Real &v = D(i);
            assert(v >= 0);
            using std::sqrt;
            x.emplace_back(0, sqrt(v));
        }
        OutVector out(n);
        for (std::size_t i = 0; i < n; ++i) {
            UType &y = out[i];
            for (std::size_t j = 0; j < i; ++j) { // L is lower triangular
                y += L(i, j) * x[j];
            }
            y += x[i]; // L(i, i) == 1 with LDLT decomposition
        }
        const auto &perm = ldlt.transpositionsP().indices();
        for (std::size_t i = n; i > 0; --i) {
            using std::swap;
            swap(out[i - 1], out[perm(i - 1)]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            out[i] += mu[i];
        }
        return out;
    }
}

#endif /* end of include guard: UNCERTAINTIES_UREALS_HPP_61EB1909 */
