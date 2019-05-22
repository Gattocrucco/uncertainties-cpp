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
        Real compute_mom(const HessGrad<Real> &hg, const int n) {
            assert(n >= 1 and n <= 2);
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using ConstTriIt = typename HessGrad<Real>::ConstTriIt;
            using Diag = typename HessGrad<Real>::Diag;
            Real m(0);
            if (n == 1) {
                const ConstDiagIt dend = hg.cdend();
                for (ConstDiagIt it = hg.cdbegin(); it != dend; ++it) {
                    const Diag &d = (*it).second;
                    m += d.hhess * v<2>(d.mom);
                }
            } else if (n == 2) {
                // formula:
                // C[y^2] =
                // G_i^2 V_{ii} +
                // 2 G_i H_{ii} V_{iii} +
                // H_{ii} H_{ii} V_{iiii} +
                // 2 \sum_{i < j} (H_{ii} H_{jj} + 2 H_{ij}^2) V_{ii} V_{jj}
                for (ConstDiagIt it = hg.cdbegin(); it != hg.cdend(); ++it) {
                    const Diag &d = (*it).second;
                    m += d.grad * d.grad * v<2>(d.mom);
                    m += 2 * d.grad * d.hhess * v<3>(d.mom);
                    m += d.hhess * d.hhess * v<4>(d.mom);
                }
                
                const ConstTriIt tend = hg.ctend();
                for (ConstTriIt it = hg.ctbegin(false); it != tend; ++it) {
                    const Diag &d1 = it.diag1();
                    const Diag &d2 = it.diag2();
                    const Real &hhess = *it;
                    m += 2 * (d1.hhess * d2.hhess + 2 * hhess * hhess) * v<2>(d1.mom) * v<2>(d2.mom);
                }
            }
            return m;
        }
    }
}

#endif /* end of include guard: UNCERTAINTIES_MOMENTS_HPP_D7CCD11D */
