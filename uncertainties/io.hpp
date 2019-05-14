// io.hpp
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

#ifndef UNCERTAINTIES_IO_HPP_6DDCDE20
#define UNCERTAINTIES_IO_HPP_6DDCDE20

/*! \file
\brief Defines formatting and stream operations on `UReal`s.
*/

#include <ostream>
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>

#include "core.hpp"

namespace uncertainties {
    namespace internal {
        template<typename Real>
        int exponent(const Real &x) {
            using std::floor;
            using std::log10;
            using std::abs;
            return int(floor(log10(abs(x))));
        }
        
        template<typename Real>
        Real int_mantissa(const Real &x, const int n, const int e) {
            using std::round;
            using std::pow;
            return round(x * pow(Real(10), n - 1 - e));
        }
        
        template<typename Real>
        std::string int_mantissa_string(Real x) {
            std::string s;
            using std::abs;
            using std::floor;
            for (x = abs(x); x >= 1; x = floor(x / 10)) {
                using std::fmod;
                const Real fd = fmod(x, 10);
                const int d = int(fd);
                assert(d == fd);
                s += std::to_string(d);
            }
            if (s.size() == 0) {
                s.push_back('0');
            } else {
                std::reverse(s.begin(), s.end());
            }
            return s;
        }
        
        template<typename Real>
        int naive_ndigits(const Real &x, const float n) {
            using std::log10;
            using std::abs;
            const float log10x = float(log10(abs(x)));
            using std::floor;
            const int n_int = int(floor(n));
            const float n_frac = n - n_int;
            const float log10x_frac = log10x - std::floor(log10x);
            return n_int + (log10x_frac < n_frac ? 1 : 0);
        }
        
        template<typename Real>
        int ndigits(Real *const x, const float n) {
            const int cand_ndig = naive_ndigits(*x, n);
            const int xexp = exponent(*x);
            const Real rounded_x = int_mantissa(*x, cand_ndig, xexp);
            const int ndig = naive_ndigits(rounded_x, n);
            if (ndig > cand_ndig) {
                using std::pow;
                *x = rounded_x * pow(Real(10), xexp);
            }
            return ndig;
        }
    
        template<typename Real>
        std::string mantissa(const Real &x, const int n, int *const e) {
            const Real m = int_mantissa(x, n, *e);
            std::string s = int_mantissa_string(m);
            assert(s.size() == n or s.size() == n + 1 or (m == 0 and n < 0));
            if (n >= 1 and s.size() == n + 1) {
                *e += 1;
                s.pop_back();
            }
            return s;
        }
        
        void insert_dot(std::string *s, int n, int e);
        std::string format_exp(const int e);
        
        template<typename Real>
        std::string tostring(const Real &x) {
            const int n = 6;
            int e = exponent(x);
            std::string m = mantissa(x, n, &e);
            const bool use_exp = e >= n or e < -1;
            if (use_exp) {
                insert_dot(&m, n, 0);
                m += "e" + format_exp(e);
            } else {
                insert_dot(&m, n, e);
            }
            return m;
        }
    }
    
    template<typename Number>
    std::string format(const Number &x,
                       const float errdig,
                       const std::string &sep) {
        if (errdig <= 1.0f) {
            throw std::invalid_argument("uncertainties::format: errdig <= 1.0");
        }
        const auto mu = nom(x);
        auto s = sdev(x);
        if (s == 0) {
            return internal::tostring(mu) + sep + "0";
        }
        const int sndig = internal::ndigits(&s, errdig);
        int sexp = internal::exponent(s);
        int muexp = mu != 0 ? internal::exponent(mu) : sexp - sndig - 1;
        std::string smant = internal::mantissa(s, sndig, &sexp);
        const int mundig = sndig + muexp - sexp;
        std::string mumant = internal::mantissa(mu, mundig, &muexp);
        const std::string musign = mu < 0 ? "-" : "";
        bool use_exp;
        int base_exp;
        if (mundig >= sndig) {
            use_exp = muexp >= mundig or muexp < -1;
            base_exp = muexp;
        } else {
            use_exp = sexp >= sndig or sexp < -1;
            base_exp = sexp;
        }
        if (use_exp) {
            internal::insert_dot(&mumant, mundig, muexp - base_exp);
            internal::insert_dot(&smant, sndig, sexp - base_exp);
            return "(" + musign + mumant + sep + smant + ")e" + internal::format_exp(base_exp);
        } else {
            internal::insert_dot(&mumant, mundig, muexp);
            internal::insert_dot(&smant, sndig, sexp);
            return musign + mumant + sep + smant;
        }
    }
    
    template<typename Real, typename CharT>
    std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> &stream,
                                          const UReal<Real> &x) {
        return stream << format(x);
    }
}

#endif /* end of include guard: UNCERTAINTIES_IO_HPP_6DDCDE20 */
