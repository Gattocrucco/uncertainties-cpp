#include <chrono>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <uncertainties/ureal.hpp>
#include <uncertainties/ureals.hpp>

std::vector<double> random_cov_matrix(const int n) {
    std::vector<double> m(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[n * i + j] = m[n * j + i] = (double(std::rand()) / RAND_MAX) * 2.0 - 1.0;
        }
    }
    std::vector<double> c(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[n * i + j] += m[n * i + k] * m[n * j + k];
            }
        }
    }
    return c;
}

std::vector<double> random_mu(const int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = std::rand();
    }
    return v;
}

namespace unc = uncertainties;
using unc::udouble;

int main() {
    std::srand(0);
    const int n = 50000;
    const auto u = std::chrono::steady_clock::now();
    decltype(u - u) time;
    for (int i = 0; i < n; ++i) {
        std::vector<double> cov = random_cov_matrix(2);
        std::vector<double> mu = random_mu(2);
        const auto start = std::chrono::steady_clock::now();
        std::vector<udouble> x = unc::ureals<std::vector<udouble>>(mu, cov);
        const auto end = std::chrono::steady_clock::now();
        time += end - start;
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(time);
    std::cout << us.count() << "\n";
}
