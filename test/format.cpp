#include <string>
#include <utility>
#include <stdexcept>

#include <uncertainties/ureal.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

template<typename Real, typename... Args>
void check(const Real &n, const Real &s, const std::string &str, Args &&... args) {
    const unc::UReal<Real> x(n, s);
    std::string f = unc::format(x, std::forward<Args>(args)...);
    if (f != str) {
        throw std::runtime_error("'" + f + "' != '" + str + "'");
    }
}

int main() {
    check<double>(1, 0.2, "1.00 pm 0.20", 1.5, " pm ");
    check<double>(1, 0.3, "1.00 pm 0.30", 1.5, " pm ");
    check<double>(1, 0.31, "1.00 pm 0.31", 1.5, " pm ");
    check<double>(1, 0.32, "1.0 pm 0.3", 1.5, " pm ");
    check<double>(-1, 0.34, "-1.00 pm 0.34", 2, " pm ");
    check<double>(0, 0, "0 pm 0", 2, " pm ");
    check<double>(123456, 0, "123456 pm 0", 2, " pm ");
    check<double>(12345.6, 0, "12345.6 pm 0", 2, " pm ");
    check<double>(12345.67, 0, "12345.7 pm 0", 2, " pm ");
    check<double>(1e8, 0, "1.00000e+8 pm 0", 2, " pm ");
    check<double>(1e-2, 0, "1.00000e-2 pm 0", 2, " pm ");
    check<double>(1e-1, 0, "0.100000 pm 0", 2, " pm ");
    check<double>(12345.99, 0, "12346.0 pm 0", 2, " pm ");
    check<double>(0, 0.001, "(0.0 pm 1.0)e-3", 2, " pm ");
    check<double>(0, 0.01, "(0.0 pm 1.0)e-2", 2, " pm ");
    check<double>(0, 0.1, "0.00 pm 0.10", 2, " pm ");
    check<double>(0, 1, "0.0 pm 1.0", 2, " pm ");
    check<double>(0, 10, "0 pm 10", 2, " pm ");
    check<double>(0, 100, "(0.0 pm 1.0)e+2", 2, " pm ");
    check<double>(0, 1000, "(0.0 pm 1.0)e+3", 2, " pm ");
    check<double>(0, 0.0196, "(0.0 pm 2.0)e-2", 2, " pm ");
    check<double>(0, 0.196, "0.00 pm 0.20", 2, " pm ");
    check<double>(0, 1.96, "0.0 pm 2.0", 2, " pm ");
    check<double>(0, 19.6, "0 pm 20", 2, " pm ");
    check<double>(0, 196, "(0.0 pm 2.0)e+2", 2, " pm ");
    check<double>(0, 0.00996, "(0.0 pm 1.0)e-2", 2, " pm ");
    check<double>(0, 0.0996, "0.00 pm 0.10", 2, " pm ");
    check<double>(0, 0.996, "0.0 pm 1.0", 2, " pm ");
    check<double>(0, 9.96, "0 pm 10", 2, " pm ");
    check<double>(0, 99.6, "(0.0 pm 1.0)e+2", 2, " pm ");
    check<double>(0.025, 3, "0.0 pm 3.0", 2, " pm ");
    check<double>(0.025, 0.3, "0.03 pm 0.30", 2, " pm ");
    check<double>(0.025, 0.03, "(2.5 pm 3.0)e-2", 2, " pm ");
    check<double>(0.025, 0.003, "(2.50 pm 0.30)e-2", 2, " pm ");
    check<double>(0.0025, 0.003, "(2.5 pm 3.0)e-3", 2, " pm ");
    check<double>(0.25, 3, "0.3 pm 3.0", 2, " pm ");
    check<double>(2.5, 3, "2.5 pm 3.0", 2, " pm ");
    check<double>(25, 3, "25.0 pm 3.0", 2, " pm ");
    check<double>(2500, 300, "(2.50 pm 0.30)e+3", 2, " pm ");
    check<double>(1, 0.99, "1.0 pm 1.0", 1.5, " pm ");
}
