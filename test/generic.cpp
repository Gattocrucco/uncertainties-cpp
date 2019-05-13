#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

#include <uncertainties/stat.hpp>
#include <uncertainties/io.hpp>
#include <uncertainties/math.hpp>
#include <uncertainties/ureal.hpp>

namespace unc = uncertainties;
using unc::udouble;
using unc::ufloat;

template<typename Vector>
void print_matrix(Vector m) {
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

int main() {
    udouble x(1, 1);
    udouble y {1, 1};
    std::cout << "x - x = " << (x - x).format() << "\n";
    std::cout << "x + x = " << (x + x).format() << "\n";
    std::cout << "x - y = " << (x - y).format() << "\n";
    std::cout << "sizeof(udouble) = " << sizeof(udouble) << "\n";
    std::cout << "sizeof(ufloat) = " << sizeof(ufloat) << "\n";
    std::cout << "ufloat(x) = " << ufloat(x).format() << "\n";
    std::cout << "x - ufloat(x) = " << (x - udouble(ufloat(x))).format() << "\n";
    std::cout << "cov(x, x) = " << cov(x, x) << "\n";
    std::cout << "cov(x, -x) = " << cov(x, -x) << "\n";
    std::cout << "cov(x, y) = " << cov(x, y) << "\n";
    std::cout << "cov(x, x + y) = " << cov(x, x + y) << "\n";
    x += 1;
    std::cout << "x += 1; x = " << x.format() << "\n";
    x += x;
    std::cout << "x += x; x = " << x.format() << "\n";
    std::cout << "x * x = " << (x * x).format() << "\n";
    x *= x;
    std::cout << "x *= x; x = " << x.format() << "\n";
    std::vector<double> corr = unc::corr_matrix<std::vector<double>>(std::vector<udouble>{x, 2 * x, y, x + y});
    std::cout << "corr_matrix({x, 2 * x, y, x + y}) = \n";
    print_matrix(corr);
    std::vector<udouble> v {x, 2 * x, y, x + y};
    std::vector<double> cov(v.size() * v.size());
    unc::cov_matrix(v.begin(), v.end(), cov.begin());
    std::cout << "cov_matrix({x, 2 * x, y, x + y}) = \n";
    print_matrix(cov);
}
