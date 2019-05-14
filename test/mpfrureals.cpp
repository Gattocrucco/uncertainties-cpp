#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <limits>

#include <boost/multiprecision/mpfr.hpp>

namespace mp = boost::multiprecision;
using type = mp::number<mp::mpfr_float_backend<20>, mp::et_off>;

#include <uncertainties/ureal.hpp>
#include <uncertainties/ureals.hpp>
#include <uncertainties/stat.hpp>

namespace unc = uncertainties;

using utype = unc::UReal<type>;

std::vector<type> random_cov_matrix(const int n) {
    std::vector<type> m(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[n * i + j] = m[n * j + i] = (type(std::rand()) / RAND_MAX) * 2.0 - 1.0;
        }
    }
    std::vector<type> c(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[n * i + j] += m[n * i + k] * m[n * j + k];
            }
        }
    }
    return c;
}

std::vector<type> random_mu(const int n) {
    std::vector<type> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = std::rand();
    }
    return v;
}

bool close(const type &x, const type &y, const type &atol=1e-8, const type &rtol=1e-8) {
    using std::abs;
    return abs(x - y) < abs(x + y) * rtol + atol;
}

bool allclose(const std::vector<type> &v, const std::vector<type> &w) {
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

bool check(int n) {
    const std::vector<type> cov = random_cov_matrix(n);
    const std::vector<type> mu = random_mu(n);
    const std::vector<utype> x = unc::ureals<std::vector<utype>>(mu, cov);
    const std::vector<type> c_cov = unc::cov_matrix<std::vector<type>>(x);
    const std::vector<type> c_mu = unc::nom_vector<std::vector<type>>(x);
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
