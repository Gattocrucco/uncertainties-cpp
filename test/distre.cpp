#include <stdexcept>
#include <sstream>
#include <initializer_list>
#include <cmath>
#include <ios>
#include <iomanip>
#include <vector>
#include <array>
#include <fstream>
#include <random>

#include <uncertainties/ureal2.hpp>
#include <uncertainties/distr.hpp>
#include <uncertainties/io.hpp>

namespace unc = uncertainties;

template<typename Real>
bool close(const Real &x, const Real &y, const Real &atol=1e-8, const Real &rtol=1e-8) {
    using std::abs;
    return abs(x - y) <= atol + (abs(x) + abs(y)) * rtol;
}

template<typename Real, unc::Prop prop>
void checkmn(const unc::UReal2<Real, prop> &x, const Real &mn, const int n) {
    Real m = x.m(n);
    if (n == 1) {
        m += x.first_order_n();
    }
    if (not close(mn, m)) {
        std::ostringstream s;
        s << "expected m(" << n << ") = " << mn << ", got " << m;
        s << ", with x = " << x;
        throw std::runtime_error(s.str());
    }
    if (n == 2) {
        const Real c = cov(x, x);
        if (not close(c, m)) {
            std::ostringstream s;
            s << std::setprecision(15) << std::scientific;
            s << "var(x) = " << m << " but cov(x,x) = " << c;
            s << ", with x = " << x;
            throw std::runtime_error(s.str());
        }
    }
}

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &x, const std::initializer_list<Real> &mlist) {
    int i = 0;
    for (const auto &m : mlist) {
        checkmn(x, m, ++i);
    }
}

template<typename Real>
std::array<Real, 9> chisquare_moments(const int k) {
    const Real K = k;
    return {
        1                                                  ,
        K                                                  ,
        K*(K + 2)                                          ,
        K*(K + 2)*(K + 4)                                  ,
        K*(K + 2)*(K + 4)*(K + 6)                          ,
        K*(K + 2)*(K + 4)*(K + 6)*(K + 8)                  ,
        K*(K + 2)*(K + 4)*(K + 6)*(K + 8)*(K + 10)         ,
        K*(K + 2)*(K + 4)*(K + 6)*(K + 8)*(K + 10)*(K + 12),
        K*(K + 2)*(K + 4)*(K + 6)*(K + 8)*(K + 10)*(K + 12)*(K + 14)
    };
}

using namespace unc::distr;
using type = double;
using utype = unc::UReal2M<type>;
using etype = unc::UReal2E<type>;

int main() {
    // test with arbitrary moments
    // x, y independent
    // E[(x+y)^n] = \sum_{k=0}^n (n k) E[x^k] E[y^{n-k}]
    // E[(x^2)^n] = E[x^{2n}]
    
    const std::vector<type> coeffs {1, 1.24, -1.2355, 0.98};
    const std::vector<int> ks {1, 2, 3, 4};
    assert(coeffs.size() == ks.size());

    // make chisquare variables
    std::vector<utype> v;
    for (int i = 0; i < coeffs.size(); ++i) {
        v.push_back(chisquare<utype>(ks[i]));
    }
    
    // expression to check moments of
    utype r;
    for (int i = 0; i < v.size(); ++i) {
        r = r + coeffs[i] * v[i];
    }
    r = r * r;
    
    // write down zero-centered moments
    std::vector<std::array<type, 9>> zm;
    for (int i = 0; i < coeffs.size(); ++i) {
        zm.push_back(chisquare_moments<type>(ks[i]));
    }
    
    // compute moments of linear combination
    std::array<type, 9> m;
    m.fill(0);
    m[0] = 1.0;
    for (int i = 0; i < zm.size(); ++i) {
        const std::array<type, 9> &addm = zm[i];
        std::array<type, 9> newm;
        for (int n = 0; n < m.size(); ++n) {
            newm[n] = 0;
            for (int k = 0; k <= n; ++k) {
                const type mnk = addm[n - k] * std::pow(coeffs[i], n - k);
                newm[n] += unc::internal::binom_coeff(n, k) * m[k] * mnk;
            }
        }
        std::copy(newm.begin(), newm.end(), m.begin());
        assert(close(m[0], 1.0));
    }
    
    // compute moments of square
    std::array<type, 5> sm;
    for (int i = 0; i < sm.size(); ++i) {
        sm[i] = m[2 * i];
    }
    assert(close(sm[0], 1.0));
    
    // center moments around mean
    std::array<type, 5> cm;
    for (int n = 0; n < cm.size(); ++n) {
        cm[n] = 0;
        for (int k = 0; k <= n; ++k) {
            using std::pow;
            cm[n] += unc::internal::binom_coeff(n, k) * sm[k] * pow(-sm[1], n - k);
        }
    }
    assert(close(cm[0], 1.0));
    assert(close(cm[1], 0.0));

    check(r, {sm[1], cm[2], cm[3], cm[4]});
    
    // now we convert r to estimate unbiasing and save to file
    // both the true moments and a sample of moments of r-like
    
    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open("distre.txt", std::ofstream::trunc | std::ofstream::out);
    
    file << std::setprecision(15) << std::scientific;
    file << "# first line are true value and moments,\n";
    file << "# following lines are estimated moments for each sample\n";
    file << r.first_order_n() << " " << r.m(2) << " " << r.m(3) << " " << r.m(4) << "\n";
    file << "##################\n";
    
    std::mt19937 engine;
    std::vector<std::chi_squared_distribution<type>> chi2;
    for (int i = 0; i < ks.size(); ++i) {
        chi2.emplace_back(ks[i]);
    }
    
    constexpr int nsamples = 10000;
    for (int i = 0; i < nsamples; ++i) {
        etype er;
        for (int j = 0; j < ks.size(); ++j) {
            const type sample = chi2[j](engine);
            const etype x = chisquare<etype>(ks[j]);
            er = er + coeffs[j] * (x - x.first_order_n() + sample);
        }
        er = er * er;
        file << er.n() << " " << er.m(2) << " " << er.m(3) << " " << er.m(4) << "\n";
    }
}
