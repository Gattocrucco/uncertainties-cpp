#ifndef UNCERTAINTIES_UREALS_HPP_61EB1909
#define UNCERTAINTIES_UREALS_HPP_61EB1909

/*! \file
\brief Defines function `ureals` to generate a list of variables with given
covariance matrix.
*/

#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <string>

#include <Eigen/Dense>

#include "core.hpp"

namespace uncertainties {
    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector ureals(const InVectorA &mu,
                     const InVectorB &cov,
                     const Order order) {
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
        // can be optimized in the following ways:
        // - use Eigen::Map to directly use cov.data()
        // - specify storage order with Eigen template parameter
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                V(i, j) = order == Order::row_major ? cov[n * i + j] : cov[n * j + i];
            }
        }
        // will this throw if V is not self-adjoint? answer: no
        Eigen::SelfAdjointEigenSolver<Matrix> solver(V);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "uncertainties::ureals: error diagonalizing covariance matrix"
            );
        }
        Matrix U = solver.eigenvectors();
        Vector var = solver.eigenvalues();
        // the following can be optimized by using UReal internals
        OutVector x;
        for (std::size_t i = 0; i < n; ++i) {
            const Real v = var(i);
            if (v < 0) {
                throw std::invalid_argument(
                    "uncertainties::ureals: covariance matrix has "
                    "negative eigenvalue " + std::to_string(v)
                );
            }
            x.push_back(UType(0, std::sqrt(v)));
        }
        OutVector out;
        for (std::size_t i = 0; i < n; ++i) {
            UType y(mu[i]);
            for (std::size_t j = 0; j < n; ++j) {
                y += U(i, j) * x[j];
            }
            out.push_back(std::move(y));
        }
        return out;
    }
}

#endif /* end of include guard: UNCERTAINTIES_UREALS_HPP_61EB1909 */
