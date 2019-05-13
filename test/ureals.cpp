#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <uncertainties/ureal.hpp>
#include <uncertainties/ureals.hpp>
#include <uncertainties/io.hpp>
#include <uncertainties/stat.hpp>

template<typename Vector>
void print_matrix(const Vector &m) {
    const int n = static_cast<int>(std::round(std::sqrt(m.size())));
    char s[1024];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::snprintf(s, sizeof(s), "%10.6g, ", double(m[n * i + j]));
            std::cout << s;
        }
        std::cout << "\n";
    }
}

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

std::vector<double> random_mu(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = std::rand();
    }
    return v;
}

std::vector<double> random_sigma(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = 0.1 + double(std::rand()) / RAND_MAX;
    }
    return v;
}

bool close(double x, double y, double atol=1e-8, double rtol=1e-8) {
    return std::abs(x - y) < std::abs(x + y) * rtol + atol;
}

bool allclose(std::vector<double> &v, std::vector<double> &w) {
    if (v.size() != w.size()) {
        throw std::invalid_argument("allclose: v.size() != w.size()");
    }
    for (int i = 0; i < v.size(); ++i) {
        if (not close(v[i], w[i])) {
            return false;
        }
    }
    return true;
}

namespace unc = uncertainties;
using unc::udouble;

bool check(int n) {
    std::vector<double> cov = random_cov_matrix(n);
    std::vector<double> mu = random_mu(n);
    std::vector<double> sigma = random_sigma(n);
    std::vector<udouble> x = unc::ureals<std::vector<udouble>>(mu, cov);
    std::vector<double> c_cov = unc::cov_matrix<std::vector<double>>(x);
    std::vector<double> c_mu = unc::nom_vector<std::vector<double>>(x);
    return allclose(c_mu, mu) and allclose(cov, c_cov);
}

int main() {
    std::srand(0);
    for (int i = 0; i < 30; ++i) {
        bool ok = check(i);
        if (not ok) {
            throw std::runtime_error("problem at i = " + std::to_string(i));
        }
    }
}