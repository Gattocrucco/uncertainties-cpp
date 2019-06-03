// moments.hpp
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

#ifndef UNCERTAINTIES_MOMENTS_HPP_D7CCD11D
#define UNCERTAINTIES_MOMENTS_HPP_D7CCD11D

#include <cassert>

#include "hessgrad.hpp"

namespace uncertainties {
    namespace internal {
        template<typename Real>
        Real compute_m1(const HessGrad<Real> &hg) {
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using Diag = typename HessGrad<Real>::Diag;
            Real m(0);
            const ConstDiagIt dend = hg.cdend();
            for (ConstDiagIt it = hg.cdbegin(); it != dend; ++it) {
                const Diag &d = it->second;
                m += d.hhess;
            }
            return m;
        }

        template<typename Real>
        Real compute_m2(const HessGrad<Real> &hg) {
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using ConstTriIt = typename HessGrad<Real>::ConstTriIt;
            using Diag = typename HessGrad<Real>::Diag;
            Real m(0);
            
            // formula:
            // C[y^2] =
            // G_i^2 V_{ii} +
            // 2 G_i H_{ii} V_{iii} +
            // H_{ii} H_{ii} V_{iiii} +
            // 2 \sum_{i < j} (H_{ii} H_{jj} + 2 H_{ij}^2) V_{ii} V_{jj}
            
            for (ConstDiagIt it = hg.cdbegin(); it != hg.cdend(); ++it) {
                const Diag &d = (*it).second;
                m += d.grad * d.grad;
                m += 2 * d.grad * d.hhess * v<3>(d.mom);
                m += d.hhess * d.hhess * v<4>(d.mom);
            }
            
            const ConstTriIt tend = hg.ctend();
            for (ConstTriIt it = hg.ctbegin(false); it != tend; ++it) {
                const Diag &d1 = it.diag1();
                const Diag &d2 = it.diag2();
                const Real &hhess = *it;
                m += 2 * (d1.hhess * d2.hhess + 2 * hhess * hhess);
            }
            
            assert(m >= 0);
            return m;
        }

        template<typename Real>
        Real compute_m3(const HessGrad<Real> &hg) {
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using ConstTriIt = typename HessGrad<Real>::ConstTriIt;
            using Diag = typename HessGrad<Real>::Diag;
            Real m(0);
            
            // formula:
            // A1)  G_i G_i G_i V_iii + 
            // A2)  3 G_i G_i H_ii V_iiii + 
            // B1)  3 G_j G_j H_ii V_iijj + 
            // D1)  6 G_i G_j H_ij V_iijj + 
            // A3)  3 G_i H_ii H_ii V_iiiii + 
            // D2)  12 G_j H_ii H_ij V_iiijj + 
            // D3)  12 G_i H_ij H_ij V_iiijj + 
            // B2)  6 G_i H_ii H_jj V_iiijj + 
            // A4)  H_ii H_ii H_ii V_iiiiii + 
            // D4)  12 H_ii H_ij H_ij V_iiiijj + 
            // B3)  3 H_ii H_ii H_jj V_iiiijj + 
            // D5)  4 H_ij H_ij H_ij V_iiijjj + 
            // D6)  6 H_ii H_ij H_jj V_iiijjj + 
            // E1)  8 H_ij H_ik H_jk V_iijjkk + 
            // F1)  6 H_ii H_jk H_jk V_iijjkk + 
            // C1)  H_ii H_jj H_kk V_iijjkk
            
            const ConstDiagIt dend = hg.cdend();
            const ConstTriIt tend = hg.ctend();

            // CYCLE A
            for (ConstDiagIt it = hg.cdbegin(); it != dend; ++it) {
                const Real &grad = it->second.grad;
                const Real &hhess = it->second.hhess;
                const Moments<Real> &mom = it->second.mom;
                m += grad * grad * grad * v<3>(mom); // A1
                m += 3 * grad * grad * hhess * v<4>(mom); // A2
                m += 3 * grad * hhess * hhess * v<5>(mom); // A3
                m += hhess * hhess * hhess * hhess * v<6>(mom); // A4
                
                // CYCLE B
                ConstDiagIt it2 = it;
                for (++it2; it2 != dend; ++it2) {
                    const Real &grad2 = it2->second.grad;
                    const Real &hhess2 = it2->second.hhess;
                    const Moments<Real> &mom2 = it2->second.mom;
                    m += 3 * grad2 * grad2 * hhess; // B1(i, j)
                    m += 3 * grad * grad * hhess2; // B1(j, i)
                    m += 6 * grad * hhess * hhess2 * v<3>(mom); // B2(i, j)
                    m += 6 * grad2 * hhess2 * hhess * v<3>(mom2); // B2(j, i)
                    m += 3 * hhess * hhess * hhess2 * v<4>(mom); // B3(i, j)
                    m += 3 * hhess2 * hhess2 * hhess * v<4>(mom2); // B3(j, i)
                    
                    // CYCLE C
                    ConstDiagIt it3 = it2;
                    for (++it3; it3 != dend; ++it3) {
                        const Real &grad3 = it3->second.grad;
                        const Real &hhess3 = it3->second.hhess;
                        m += 6 * hhess * hhess2 * hhess3; // C1
                    }
                }
                
                // CYCLE F
                // This cycle could be more efficient by starting from
                // `it` using map::lower_bound if the off-diagonal terms in
                // `hg` were in a separate map.
                for (ConstTriIt it2 = hg.ctbegin(true); it2 != tend; ++it2) {
                    if (it2.id1() == it->first or it2.id2() == it->first) {
                        continue;
                    }
                    const Real &hhess2 = *it2;
                    m += 2 * 6 * hhess * hhess2 * hhess2; // F1
                }
            }
            
            // CYCLE D
            for (ConstTriIt it = hg.ctbegin(true); it != tend; ++it) {
                const Diag &d1 = it.diag1();
                const Diag &d2 = it.diag2();
                const Real hhess = *it;
                m += 2 * 6 * d1.grad * d2.grad * hhess; // D1
                m += 12 * d2.grad * d1.hhess * hhess * v<3>(d1.mom); // D2(i, j)
                m += 12 * d1.grad * d2.hhess * hhess * v<3>(d2.mom); // D2(j, i)
                m += 12 * d1.grad * hhess * hhess * v<3>(d1.mom); // D3(i, j)
                m += 12 * d2.grad * hhess * hhess * v<3>(d2.mom); // D3(j, i)
                m += 12 * d1.hhess * hhess * hhess * v<4>(d1.mom); // D4(i, j)
                m += 12 * d2.hhess * hhess * hhess * v<4>(d2.mom); // D4(j, i)
                m += 2 * 4 * hhess * hhess * hhess * v<3>(d1.mom) * v<3>(d2.mom); // D5
                m += 2 * 6 * d1.hhess * hhess * d2.hhess * v<3>(d1.mom) * v<3>(d2.mom); // D6
                
                // CYCLE E
                ConstTriIt it2 = it;
                for (++it2; it2 != tend && it2.id1() == it.id1(); ++it2) {
                    // it = ij
                    // it2 = ik
                    // and jk manually
                    const Real hhess2 = *it2;
                    const Real hhess3 = hg.tri_get(it.id2(), it2.id2());
                    m += 6 * 8 * hhess * hhess2 * hhess3; // E1
                }
            }
            return m;
        }

        template<typename Real>
        Real compute_mom(const HessGrad<Real> &hg, const int n) {
            assert(n >= 1 and n <= 3);
            switch (n) {
                case 1:
                return compute_m1(hg);
                case 2:
                return compute_m2(hg);
                case 3:
                return compute_m3(hg);
                default:
                return 0;
            }
        }
        
        template<typename Real>
        Real compute_c2(const HessGrad<Real> &hga, const HessGrad<Real> &hgb) {
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using ConstTriIt = typename HessGrad<Real>::ConstTriIt;
            using Diag = typename HessGrad<Real>::Diag;
            
            // formula:
            // A1)  G(a)_i G(b)_i V_ii + 
            // A2)  H(a)_ii G(b)_i V_iii + 
            // A3)  G(a)_i H(b)_ii V_iii + 
            // A4)  H(a)_ii H(b)_ii V_iiii + 
            // B1)  2 H(a)_ij H(b)_ij V_iijj +
            // C1)  H(a)_ii H(b)_jj V_iijj
            
            Real m(0);
            
            // CYCLE A
            const ConstDiagIt denda = hga.cdend();
            const ConstDiagIt dendb = hgb.cdend();
            ConstDiagIt ita = hga.cdbegin();
            ConstDiagIt itb = hgb.cdbegin();
            while (ita != denda and itb != dendb) {
                const Id ida = ita->first;
                const Id idb = itb->first;
                if (ida == idb) {
                    const Diag &da = ita->second;
                    const Diag &db = itb->second;
                    m += da.grad * db.grad; // A1
                    m += da.hhess * db.grad * v<3>(da.mom); // A2
                    m += da.grad * db.hhess * v<3>(da.mom); // A3
                    m += da.hhess * db.hhess * v<4>(da.mom); // A4
                }
                if (ida <= idb) ++ita;
                if (ida <= idb) ++itb;
            }
            
            // CYCLE B
            const ConstTriIt tenda = hga.ctend();
            const ConstTriIt tendb = hgb.ctend();
            ConstTriIt tita = hga.ctbegin(true);
            ConstTriIt titb = hgb.ctbegin(true);
            while (tita != tenda and titb != tendb) {
                const std::pair<Id, Id> ida = tita.id();
                const std::pair<Id, Id> idb = titb.id();
                if (ida == idb) {
                    m += 2 * 2 * (*tita) * (*titb); // B1
                }
                if (ida <= idb) ++tita;
                if (ida <= idb) ++titb;
            }
            
            // CYCLE C
            // Could be made more efficient with a find on the hgb nodes.
            for (ita = hga.cdbegin(); ita != denda; ++ita) {
                for (itb = hgb.cdbegin(); itb != dendb; ++itb) {
                    if (ita->first != itb->first) {
                        const Diag &da = ita->second;
                        const Diag &db = itb->second;
                        m += da.hhess * db.hhess; // C1
                    }
                }
            }
            
            return m;
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_MOMENTS_HPP_D7CCD11D */
