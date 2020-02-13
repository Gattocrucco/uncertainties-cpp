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


#ifndef UNCERTAINTIES_CHECKMOMENTS_HPP_502A3413
#define UNCERTAINTIES_CHECKMOMENTS_HPP_502A3413

#include <array>
#include <cassert>
#include <sstream>

#include <Eigen/Dense>

// This files defines the function `check_moments_throw` used by UReal2 when
// constructing to check validity of moments. I've read the way to check
// moments on wikipedia: https://en.wikipedia.org/wiki/Hamburger_moment_problem.
// There they define it for the complete infinite list of moments, while I use
// it on a submatrix. I've not even checked theoretically if this should work,
// I have just done a lot of empirical testing (besides the tests hardcoded in
// test/checkmom.cpp).

namespace uncertainties {
    namespace internal {        
        template<typename Real>
        using HKMatrix = Eigen::Matrix<Real, 5, 5>;
        // "HK" stands for Henkel Kernel.
       
        // Return 0 if the matrix is positive semidefinite. Return 1 if it is
        // negative definite. Otherwise, return min_eigenvalue / max_eigenvalue.
        // Note this last formula will be negative, think of min_eigenvalue as
        // "max negative eigenvalue". So: 0 = ok, > 0 = definitely not ok,
        // < 0 = probably not ok, check that the number is small enough.
        //
        // We need to check if the matrix is positive semidefinite, and this can
        // be faster than computing all the eigenvalues. However I think this
        // is low priority since the matrix is 5x5. A possibly faster solution
        // I already know is using Eigen's LDLT decomposition, but I would not
        // know how to produce the thresholdable output with that.
        template<typename Real>
        Real hamburger(const HKMatrix<Real> &M) {
            Eigen::SelfAdjointEigenSolver<HKMatrix<Real>> solver(M, Eigen::EigenvaluesOnly);
            if (solver.info() != Eigen::Success) {
                std::ostringstream ss;
                ss << "uncertainties::internal::hamburger: diagonalization failed,";
                ss << " maybe you passed nonfinite moments to an UReal2 constructor?";
                throw std::runtime_error(ss.str());
            }
            const Eigen::Matrix<Real, 5, 1> v = solver.eigenvalues();
            Real max_nonneg = -1;
            Real max_neg = 1;
            for (int i = 0; i < v.size(); ++i) {
                const Real &a = v(i);
                if (a < 0 and a < max_neg) {
                    max_neg = a;
                } else if (a >= 0 and a > max_nonneg) {
                    max_nonneg = a;
                }
            }
            if (max_nonneg < 0) {
                // no nonnegative eigenvalue, so they are all negative
                return 1;
            } else if (max_neg > 0) {
                // no negative eigenvalue, so they are all nonnegative
                return 0;
            } else {
                return max_neg / max_nonneg;
            }
        }
        
        // Given an array of central moments 2 to 8, or standardized moments
        // 3 to 8 (the length of the array argument decides that), return 0
        // if the moments are realizable (i.e. there exists a distribution with
        // those moments), otherwise a positive number if they are surely
        // irrealizable, a negative one if the result may be unsure due to
        // numerical error; in the latter case the number then has to be
        // compared to a threshold (the closer to zero, the better).
        template<typename Real, std::size_t n>
        Real check_moments(const std::array<Real, n> &moments) {
            static_assert(n == 6 or n == 7, "n != 6 and 7");
            HKMatrix<Real> M;
            constexpr int offset = n - 6;
            for (int j = 0; j < 5; ++j) {
                for (int i = j; i < 5; ++i) {
                    const int idx = i + j - 3 + offset;
                    M(i, j) = idx >= 0 ? moments[idx] : std::abs((idx + offset) % 2);
                }
            }
            return hamburger(M);
        }
        
        // Given an array of central moments 2 to 8, or standardized moments
        // 3 to 8 (the length of the array argument decides that), throw a
        // std::invalid_argument exception if the moments are not realizable.
        // Threshold is a small nonnegative threshold for numerical error.
        // Numerical close ties should happen only when the moments are on the
        // border of allowed moments, so to be conservative you can pass 0 to
        // nail "dangerous but allowed" moments.
        template<typename Real, std::size_t n>
        void check_moments_throw(const std::array<Real, n> & moments,
                                 const Real &threshold) {
            static_assert(n == 6 or n == 7, "n != 6 and 7");
            assert(threshold >= 0);
            const Real cond = check_moments(moments);
            if (cond < -threshold or cond > 0) {
                std::ostringstream ss;
                ss << "uncertainties::UReal2::UReal2: ";
                if (n == 6) {
                    ss << "the array of standardized central moments [";
                } else {
                    ss << "the array of central moments [";
                }
                ss << moments[0] << ", ";
                ss << moments[1] << ", ";
                ss << moments[2] << ", ";
                ss << moments[3] << ", ";
                ss << moments[4] << ", ";
                if (n == 6) {
                    ss << moments[5] << "] ";
                    ss << "(moments from third to eighth) ";
                } else {
                    ss << moments[5] << ", ";
                    ss << moments[6] << "] ";
                    ss << "(moments from second to eighth) ";
                }
                ss << "does not represent a valid probability distribution; ";
                if (cond < 0) {
                    ss << "the ``condition number'' of the Henkel kernel ";
                    ss << "(minimum negative eigenvalue over maximum positive eigenvalue) ";
                    ss << "is " << cond << ", which is less than the required ";
                    ss << "threshold " << -threshold << ". ";
                } else {
                    ss << "all the eigenvalues of the Henkel kernel are negative. ";
                }
                ss << "See uncertainties-cpp documentation and ";
                ss << "https://en.wikipedia.org/wiki/Hamburger_moment_problem";
                throw std::invalid_argument(ss.str());
            }
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_CHECKMOMENTS_HPP_502A3413 */
