// checkmoments.hpp
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


#ifndef CHECKMOMENTS_HPP_502A3413
#define CHECKMOMENTS_HPP_502A3413

#include <array>
#include <cstdlib>

#include <Eigen/Dense>

namespace uncertainties {
    namespace internal {
        template<typename Real>
        bool check_std_moments(const std::array<Real, 6> &std_moments) {
            using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
            Matrix M(5, 5);
            for (int j = 0; j < 5; ++j) {
                for (int i = j; i < 5; ++i) {
                    const int idx = i + j - 3;
                    M(i, j) = idx >= 0 ? std_moments[idx] : std::abs(idx % 2);
                }
            }
            Eigen::LDLT<Matrix> ldlt(M);
            return ldlt.isPositive();
        }

        template<typename Real>
        bool check_moments(const std::array<Real, 7> &moments) {
            using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
            Matrix M(5, 5);
            for (int j = 0; j < 5; ++j) {
                for (int i = j; i < 5; ++i) {
                    const int idx = i + j - 2;
                    M(i, j) = idx >= 0 ? moments[idx] : std::abs((idx + 1) % 2);
                }
            }
            Eigen::LDLT<Matrix> ldlt(M);
            return ldlt.isPositive();
        }
    }
}

#endif /* end of include guard: CHECKMOMENTS_HPP_502A3413 */
