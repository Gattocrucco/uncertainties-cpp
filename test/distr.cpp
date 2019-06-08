#include <stdexcept>
#include <sstream>
#include <initializer_list>
#include <cmath>
#include <ios>
#include <iomanip>
#include <vector>
#include <array>

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
void checkcov(const unc::UReal2<Real, prop> &x, const unc::UReal2<Real, prop> &y, const Real &c) {
    const Real cov1 = cov(x, y);
    const Real cov2 = cov(y, x);
    if (not close(c, cov1)) {
        std::ostringstream s;
        s << "expected cov(x,y) = " << c << ", got " << cov1;
        s << ", with x = " << x << ", y = " << y;
        throw std::runtime_error(s.str());
    }
    if (not close(cov1, cov2)) {
        std::ostringstream s;
        s << "cov(x,y) = " << cov1 << " but cov(y,x) = " << cov2;
        s << ", with x = " << x << ", y = " << y;
        throw std::runtime_error(s.str());
    }
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

template<typename Real, unc::Prop prop>
void check(const unc::UReal2<Real, prop> &x, const unc::UReal2<Real, prop> &y) {
    for (int i = 1; i <= 4; ++i) {
        const Real xm = i == 1 ? x.n() : x.m(i);
        const Real ym = i == 1 ? y.n() : y.m(i);
        if (not close(xm, ym)) {
            std::ostringstream ss;
            ss << "x.m(" << i << ") = " << xm << " != y.m(";
            ss << i << ") = " << ym;
            ss << ", with x = " << x << ", y = " << y;
            throw std::runtime_error(ss.str());
        }
    }
}

using namespace unc::distr;
using utype = unc::udouble2m;
using type = utype::real_type;

int main() {
    // normal
    check(normal<utype>(1, 1), {1.0, 1.0, 0.0, 3.0});
    check(normal<utype>(1, 2), {1.0, 4.0, 0.0, 48.0});
    
    // sum of normal is normal
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            const type v = i * i + j * j;
            check(normal<utype>(0, i) + normal<utype>(0, j), {0.0, v, 0.0, 3 * v * v});
        }
    }
    
    // linear tests on normal
    utype x = normal<utype>(1, 1);
    check(x - x, {0.0, 0.0, 0.0, 0.0});
    check(x + x, {2.0, 4.0, 0.0, 48.0});
    check(2 * x, {2.0, 4.0, 0.0, 48.0});
    check(x + 2 * x, {3.0, 9.0, 0.0, 243.0});
    check(-x, {-1.0, 1.0, 0.0, 3.0});
    checkcov(x, x, 1.0);
    checkcov(x, utype(1.0), 0.0);
    checkcov(x + x, x - x, 0.0);

    // chisquare
    check(chisquare<utype>(0), {0.0, 0.0, 0.0, 0.0});
    check(chisquare<utype>(1), {1.0, 2.0, 8.0, 60.0});
    check(chisquare<utype>(2), {2.0, 4.0, 16.0, 144.0});
    
    // sum of chisquare is chisquare
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            const type k = i + j;
            check(chisquare<utype>(i) + chisquare<utype>(j), {k, 2 * k, 8 * k, 12 * k * (k + 4)});
        }
    }
    
    // sum of normal squared is chisquare
    x = normal<utype>(0, 1);
    check(x * x, chisquare<utype>(1));
    for (int i = 0; i < 10; ++i) {
        utype x;
        for (int k = 0; k < i - 1; ++k) {
            utype n = normal<utype>(0, 1);
            x = x + n * n;
        }
        utype shared_n = normal<utype>(0, 1);
        if (i > 0) {
            x = x + 0.5 * shared_n * shared_n;
        }
        for (int j = 0; j < 10; ++j) {
            utype y;
            for (int k = 0; k < j; ++k) {
                utype n = normal<utype>(0, 1);
                y = y + n * n;
            }
            if (i > 0) {
                y = y + 0.5 * shared_n * shared_n;
            }
            const utype::real_type k = i + j;
            check(x + y, {k, 2 * k, 8 * k, 12 * k * (k + 4)});
            check(-(x + y), {-k, 2 * k, -8 * k, 12 * k * (k + 4)});
            check(2 * (x + y), {2 * k, 4 * 2 * k, 8 * 8 * k, 16 * 12 * k * (k + 4)});
            checkcov(x, y, i > 0 ? 0.5 : 0.0);
        }
    }
    
    // uniform
    using std::sqrt;
    check(uniform<utype>(0, 1), {0.5, type(1) / 12, 0.0, type(1) / 80});
    check(uniform<utype>(-1, 1), {0.0, type(1) / 3, 0.0, type(1) / 5});
    
    // covariance is bilinear
    const std::vector<utype> dists = {
        normal<utype>(0, 1),
        normal<utype>(-2.3, 0.3),
        chisquare<utype>(2),
        chisquare<utype>(10),
        uniform<utype>(0, 1),
        uniform<utype>(-1.1, 3.5)
    };
    for (int i = 0; i < dists.size(); ++i) {
        for (int j = i; j < dists.size(); ++j) {
            const utype &x = dists[i];
            const utype &y = dists[j];
            for (double alpha = -0.3; alpha < 10.0; alpha += 1.3) {
                for (double beta = -3.4; beta < 5.0; beta += 1.2) {
                    const type expcov = var(x) + (alpha + beta) * cov(x, y) + alpha * beta * var(y);
                    checkcov(x + alpha * y, y * beta + x, expcov);
                }
            }
        }
    }
    
    // create two variables with nontrivial intersection
    x = normal<utype>(-0.1234, 1.345);
    utype y = normal<utype>(0.9, 0.3849);
    utype z = normal<utype>(0.0001, 0.77);
    utype xpy = x + y;
    utype ypz = y + z;
    checkcov(xpy, ypz, var(y));
    utype tot = xpy + ypz;
    utype tot2 = x + 2 * y + z;
    check(tot, tot2);
    
    // test with arbitrary moments
    // x, y independent
    // E[(xy)^n] = \int dx dy p(x) p(y) x^n y^n = E[x^n] E[y^n]
    // E[(x+y)^n] = \sum_{k=0}^n (n k) E[x^k] E[y^{n-k}]
    // E[(x^2)^n] = E[x^{2n}]
    
    // mean, stddev, standardized moments 3 to 8
    const std::vector<type> mu {1.345, -4.32, 1.2, 0.5};
    const std::vector<type> sigma {0.56, 0.88, 2.33, 0.9};
    const std::vector<std::array<type, 6>> moments {
        {-4.203979006661760e-01,  2.306106586517433e+00, -2.401538376980557e+00,  7.714387209802071e+00, -1.227409781769712e+01,  3.230234798950539e+01},
        { 5.656854249492382e-01,  2.400000000000000e+00,  3.232488142567076e+00,  8.857142857142861e+00,  1.697056274847715e+01,  4.160000000000002e+01},
        { 2.393417858997477e-01,  1.862753541523876e+00,  1.043117800515642e+00,  4.312283112177668e+00,  3.770708129670418e+00,  1.130784371593390e+01},
        { 4.056950772626715e-01,  3.059295089399554e+00,  3.909491946824366e+00,  1.739515608079365e+01,  4.052581111168719e+01,  1.569803094595993e+02}
    };
    // const std::vector<type> mu {0, 0, 0, 0};
    // const std::vector<type> sigma {1, 2, 3, 4};
    // const std::vector<std::array<type, 6>> moments {
    //     {0, 3, 0, 15, 0, 105},
    //     {0, 3, 0, 15, 0, 105},
    //     {0, 3, 0, 15, 0, 105},
    //     {0, 3, 0, 15, 0, 105}
    // };
    assert(mu.size() == sigma.size());
    assert(mu.size() == moments.size());
    
    // make variables with given moments
    std::vector<utype> v;
    for (int i = 0; i < moments.size(); ++i) {
        v.emplace_back(mu[i], sigma[i], moments[i]);
    }
    
    // expression to check moments of
    utype r = v[0] * v[1] + v[2] * v[3];
    
    // center moments around 0 and destandardize
    std::vector<std::array<type, 9>> zm(moments.size());
    for (int i = 0; i < moments.size(); ++i) {
        std::array<type, 9> &m = zm[i];
        for (int n = 0; n < m.size(); ++n) {
            m[n] = 0;
            for (int k = 0; k <= n; ++k) {
                using std::pow;
                type mk;
                if (k == 0) mk = 1;
                else if (k == 1) mk = 0;
                else if (k == 2) mk = sigma[i] * sigma[i];
                else mk = moments[i][k - 3] * pow(sigma[i], k);
                m[n] += unc::internal::binom_coeff(n, k) * mk * pow(mu[i], n - k);
            }
        }
        assert(close(m[0], 1.0));
        assert(close(m[1], mu[i]));
    }
    
    // compute moments of products
    std::vector<std::array<type, 9>> mm(2);
    for (int i = 0; i < mm.size(); ++i) {
        for (int j = 0; j < mm[i].size(); ++j) {
            mm[i][j] = zm[2 * i][j] * zm[2 * i + 1][j];
        }
        assert(close(mm[i][0], 1.0));
    }
    
    // compute moments of sum
    std::array<type, 9> m;
    assert(m.size() <= mm[0].size());
    for (int n = 0; n < m.size(); ++n) {
        m[n] = 0;
        for (int k = 0; k <= n; ++k) {
            m[n] += unc::internal::binom_coeff(n, k) * mm[0][k] * mm[1][n - k];
        }
    }
    assert(close(m[0], 1.0));
    
    // center moments around mean
    std::array<type, 9> cm;
    for (int n = 0; n < cm.size(); ++n) {
        cm[n] = 0;
        for (int k = 0; k <= n; ++k) {
            using std::pow;
            cm[n] += unc::internal::binom_coeff(n, k) * m[k] * pow(-m[1], n - k);
        }
    }
    assert(close(cm[0], 1.0));
    assert(close(cm[1], 0.0));

    check(r, {m[1], cm[2], cm[3], cm[4]});
}
