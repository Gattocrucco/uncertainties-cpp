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
#include <Eigen/Dense>

namespace uncertainties {
    namespace internal {
        // // formulae generated with dev/hamburger.py
        //
        // template<typename Real>
        // Real check_std_moments(const std::array<Real, 6> &std_moments) {
        //     const Real &k3 = std_moments[0];
        //     const Real &k4 = std_moments[1];
        //     const Real &k5 = std_moments[2];
        //     const Real &k6 = std_moments[3];
        //     const Real &k7 = std_moments[4];
        //     const Real &k8 = std_moments[5];
        //
        //     const Real k3_2 = k3 * k3;
        //     const Real c1 = -k3_2 + k4 - 1;
        //     if (c1 < 0) return c1;
        //
        //     const Real k5_2 = k5 * k5;
        //     const Real c2 = k3*(k3*(k3_2 - 3*k4 - k6) + k5*(2*k4 + 2)) + k4*(k4*(1 - k4) + k6) - k5_2 - k6;
        //     if (c2 < 0) return c2;
        //
        //     const Real k4_2 = k4 * k4;
        //     const Real k6_2 = k6 * k6;
        //     const Real dk67 = 2 * k6 * k7;
        //     const Real k7_2 = k7 * k7;
        //     const Real c3 = k3*(k3*(k3*(k3*k8 - 2*k4*k7 - 2*k5*k6) + k4*(3*k4*k6 + 3*k5_2 - 3*k8) + k6*(k6 - k8) + k7*(2*k5 + k7)) + k4*(k4*(-4*k4*k5 + 4*k7) + k5*(-2*k6 + 2*k8) - dk67) + k5*(k5*(-2*k5 - 2*k7) + 2*k6_2 + 2*k8) - dk67) + k4*(k4*(k4*(k4_2 - 3*k6 - k8) + k5*(3*k5 + 2*k7) + k6_2 + k8) + k5*(-3*k5*k6 - 4*k7) + k6*(2*k6 + k8) - k7_2) + k5*(k5*(k5_2 + k6 - k8) + dk67) + k6*(-k6_2 - k8) + k7_2;
        //     if (c3 < 0) return c3;
        //
        //     return 0;
        // }
        //
        // template<typename Real>
        // Real check_moments(const std::array<Real, 7> &moments) {
        //     const Real &mu2 = moments[0];
        //     const Real &mu3 = moments[1];
        //     const Real &mu4 = moments[2];
        //     const Real &mu5 = moments[3];
        //     const Real &mu6 = moments[4];
        //     const Real &mu7 = moments[5];
        //     const Real &mu8 = moments[6];
        //
        //     if (mu2 < 0) return mu2;
        //
        //     const Real mu2_2 = mu2 * mu2;
        //     const Real mu3_2 = mu3 * mu3;
        //     const Real c1 = mu2*(-mu2_2 + mu4) - mu3_2;
        //     if (c1 < 0) return c1;
        //
        //     const Real mu4_2 = mu4 * mu4;
        //     const Real mu4_3 = mu4_2 * mu4;
        //     const Real mu5_2 = mu5 * mu5;
        //     const Real c2 = mu2*(mu2*(-mu2*mu6 + 2*mu3*mu5 + mu4_2) + mu4*(-3*mu3_2 + mu6) - mu5_2) + mu3*(mu3*(mu3_2 - mu6) + 2*mu4*mu5) - mu4_3;
        //     if (c2 < 0) return c2;
        //
        //     const Real mu5_3 = mu5_2 * mu5;
        //     const Real mu5_4 = mu5_2 * mu5_2;
        //     const Real mu6_2 = mu6 * mu6;
        //     const Real mu6_3 = mu6_2 * mu6;
        //     const Real mu7_2 = mu7 * mu7;
        //     const Real c3 = mu2*(mu2*(mu2*(-mu6*mu8 + mu7_2) + mu3*(2*mu5*mu8 - 2*mu6*mu7) + mu4*(mu4*mu8 - 4*mu5*mu7 + 2*mu6_2) + mu5_2*mu6) + mu3*(mu3*(-3*mu4*mu8 + 2*mu5*mu7 + mu6_2) + mu4*(4*mu4*mu7 - 2*mu5*mu6) - 2*mu5_3) + mu4*(mu4*(-3*mu4*mu6 + 3*mu5_2) + mu6*mu8 - mu7_2) + mu5*(-mu5*mu8 + 2*mu6*mu7) - mu6_3) + mu3*(mu3*(mu3*(mu3*mu8 - 2*mu4*mu7 - 2*mu5*mu6) + mu4*(3*mu4*mu6 + 3*mu5_2) - mu6*mu8 + mu7_2) + mu4*(mu5*(-4*mu4_2 + 2*mu8) - 2*mu6*mu7) + mu5*(-2*mu5*mu7 + 2*mu6_2)) + mu4*(mu4*(mu4*(mu4_2 - mu8) + 2*mu5*mu7 + mu6_2) - 3*mu5_2*mu6) + mu5_4;
        //     if (c3 < 0) return c3;
        //
        //     return 0;
        // }

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

#endif /* end of include guard: UNCERTAINTIES_CHECKMOMENTS_HPP_502A3413 */
