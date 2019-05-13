#ifndef UNCERTAINTIES_IO_HPP_6DDCDE20
#define UNCERTAINTIES_IO_HPP_6DDCDE20

/*! \file
\brief Defines string conversion and stream operations on `UReal`s.
*/

#include <ostream>
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>

#include "core.hpp"

namespace uncertainties {
    template<typename Real>
    class UReal;
    
    namespace internal {
        template<typename Real>
        int exponent(const Real &x) {
            return static_cast<int>(std::floor(std::log10(std::abs(x))));
        }
        
        template<typename Real>
        Real int_mantissa(const Real &x, const int n, const int e) {
            return std::round(x * std::pow(Real(10), n - 1 - e));
        }
    
        template<typename Real>
        int naive_ndigits(const Real &x, const float n) {
            const float log10x = static_cast<float>(std::log10(std::abs(x)));
            const int n_int = static_cast<int>(std::floor(n));
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
                *x = rounded_x * std::pow(Real(10), xexp);
            }
            return ndig;
        }
    
        template<typename Real>
        std::string mantissa(const Real &x, const int n, int *const e) {
            const long long m = static_cast<long long>(int_mantissa(x, n, *e));
            std::string s = std::to_string(std::abs(m));
            assert(s.size() == n or s.size() == n + 1 or (m == 0 and n < 0));
            if (n >= 1 and s.size() == n + 1) {
                *e += 1;
                s.pop_back();
            }
            return s;
        }
    
        void insert_dot(std::string *s, int n, int e);
        std::string format_exp(const int e);
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
            return std::to_string(mu) + sep + "0";
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
