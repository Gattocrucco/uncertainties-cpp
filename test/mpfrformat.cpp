#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/multiprecision/mpfr.hpp>

#include <uncertainties/ureal.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

using type = boost::multiprecision::mpfr_float;
using utype = unc::UReal<type>;

template<typename... Args>
void check(const type &n, const type &s, const std::string &str, Args &&... args) {
    const utype x(n, s);
    std::string f = unc::format(x, std::forward<Args>(args)...);
    if (f != str) {
        throw std::runtime_error("'" + f + "' != '" + str + "'");
    }
}

int main() {
    check(123456, 0, "123456 pm 0", 2, " pm ");
    check(12345.6, 0, "12345.6 pm 0", 2, " pm ");
    check(12345.67, 0, "12345.7 pm 0", 2, " pm ");
    check(1e8, 0, "1.00000e+8 pm 0", 2, " pm ");
    check(1e-2, 0, "1.00000e-2 pm 0", 2, " pm ");
    check(1e-1, 0, "0.100000 pm 0", 2, " pm ");
    check(12345.99, 0, "12346.0 pm 0", 2, " pm ");
    check(0, 0.001, "(0.0 pm 1.0)e-3", 2, " pm ");
    check(0, 0.01, "(0.0 pm 1.0)e-2", 2, " pm ");
    check(0, 0.1, "0.00 pm 0.10", 2, " pm ");
    check(0, 1, "0.0 pm 1.0", 2, " pm ");
    check(0, 10, "0 pm 10", 2, " pm ");
    check(0, 100, "(0.0 pm 1.0)e+2", 2, " pm ");
    check(0, 1000, "(0.0 pm 1.0)e+3", 2, " pm ");
    check(0, 0.0196, "(0.0 pm 2.0)e-2", 2, " pm ");
    check(0, 0.196, "0.00 pm 0.20", 2, " pm ");
    check(0, 1.96, "0.0 pm 2.0", 2, " pm ");
    check(0, 19.6, "0 pm 20", 2, " pm ");
    check(0, 196, "(0.0 pm 2.0)e+2", 2, " pm ");
    check(0, 0.00996, "(0.0 pm 1.0)e-2", 2, " pm ");
    check(0, 0.0996, "0.00 pm 0.10", 2, " pm ");
    check(0, 0.996, "0.0 pm 1.0", 2, " pm ");
    check(0, 9.96, "0 pm 10", 2, " pm ");
    check(0, 99.6, "(0.0 pm 1.0)e+2", 2, " pm ");
    check(0.025, 3, "0.0 pm 3.0", 2, " pm ");
    check(0.025, 0.3, "0.03 pm 0.30", 2, " pm ");
    check(0.025, 0.03, "(2.5 pm 3.0)e-2", 2, " pm ");
    check(0.025, 0.003, "(2.50 pm 0.30)e-2", 2, " pm ");
    check(0.0025, 0.003, "(2.5 pm 3.0)e-3", 2, " pm ");
    check(0.25, 3, "0.3 pm 3.0", 2, " pm ");
    check(2.5, 3, "2.5 pm 3.0", 2, " pm ");
    check(25, 3, "25.0 pm 3.0", 2, " pm ");
    check(2500, 300, "(2.50 pm 0.30)e+3", 2, " pm ");
    check(1, 0.99, "1.0 pm 1.0", 1.5, " pm ");
}
