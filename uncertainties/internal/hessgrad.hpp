// hessgrad.hpp
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

#ifndef UNCERTAINTIES_HESSGRAD_HPP_84C05D5B
#define UNCERTAINTIES_HESSGRAD_HPP_84C05D5B

#include <memory>
#include <array>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <type_traits>

#include "../core.hpp"

namespace uncertainties {
    namespace internal {
        template<typename Real>
        class Moments {
        private:
            std::array<Real, 6> m;
        
        public:
            Moments() {
                ;
            }
            
            inline explicit Moments(const std::array<Real, 6> &std_moments):
            m {std_moments} {
                ;
            }
            
            template<int n>
            inline Real &get() noexcept {
                return std::get<n>(m);
            }

            template<int n>
            inline const Real &get() const noexcept {
                return std::get<n>(m);
            }
            
            using iterator = typename std::array<Real, 6>::iterator;
            inline iterator begin() noexcept {
                return m.begin();
            }
            inline iterator end() noexcept {
                return m.end();
            }

            using const_iterator = typename std::array<Real, 6>::const_iterator;
            inline const_iterator begin() const noexcept {
                return m.begin();
            }
            inline const_iterator end() const noexcept {
                return m.end();
            }
            
            Real &operator[](int index) {
                return m[index];
            }
        };
    
        template<int n, typename Real>
        inline const Real &v(const Moments<Real> &p) noexcept {
            static_assert(n >= 3 and n <= 8, "it must be 3 <= n <= 8");
            return p.template get<n - 3>();
        }
    
        template<typename Real>
        class HessGrad {
        public:
            struct Diag {
                Real grad {0};
                Real hhess {0}; // half second derivative
                Moments<Real> mom;
            };
            
            using IdPair = std::pair<Id, Id>;
        
        private:
            std::map<Id, Diag> diagmap;
            std::map<IdPair, Real> trimap;
            
            static const Real zero;
            static const Diag zerodiag;
        
        public:
            inline typename std::map<Id, Diag>::size_type size() const noexcept {
                return diagmap.size();
            }
        
            inline const Diag &diag(const Id id) const {
                return diagmap.at(id);
            }
        
            inline Diag &diag(const Id id) {
                return diagmap[id];
            }
            
            const Diag &diag_get(const Id id) const noexcept {
                using It = typename std::map<Id, Diag>::const_iterator;
                const It it = diagmap.find(id);
                if (it != diagmap.end()) {
                    return it->second;
                } else {
                    return zerodiag;
                }
            }
            
            const Real &tri(const Id minid, const Id maxid) const {
                assert(minid < maxid);
                return trimap.at({minid, maxid});
            }

            Real &tri(const Id minid, const Id maxid) {
                assert(minid < maxid);
                return trimap[{minid, maxid}];
            }

            const Real &tri_get(const Id minid, const Id maxid) const noexcept {
                assert(minid < maxid);
                using It = typename std::map<IdPair, Real>::const_iterator;
                const It it = trimap.find({minid, maxid});
                if (it != trimap.end()) {
                    return it->second;
                } else {
                    return zero;
                }
            }
            
            inline Real &hhess(const Id minid, const Id maxid) {
                return minid != maxid ? this->tri(minid, maxid) : this->diag(minid).hhess;
            }
            
            inline const Real &hhess_get(const Id id1, const Id id2) const noexcept {
                return id1 != id2 ? this->tri_get(std::min(id1, id2), std::max(id1, id2)) : this->diag_get(id1).hhess;
            }
            
            using DiagIt = typename std::map<Id, Diag>::iterator;
        
            inline DiagIt dbegin() noexcept {
                return diagmap.begin();
            }
        
            inline DiagIt dend() noexcept {
                return diagmap.end();
            }
            
            using ConstDiagIt = typename std::map<Id, Diag>::const_iterator;

            inline ConstDiagIt cdbegin() const noexcept {
                return diagmap.cbegin();
            }
        
            inline ConstDiagIt cdend() const noexcept {
                return diagmap.cend();
            }
            
            class ConstTriIt {
            private:
                typename std::map<Id, Diag>::const_iterator diagmin;
                typename std::map<Id, Diag>::const_iterator diagmax;
                const typename std::map<Id, Diag>::const_iterator diagend;
                typename std::map<IdPair, Real>::const_iterator tri;
                const typename std::map<IdPair, Real>::const_iterator triend;
                const bool skipmissing = true;
                
                void update_diag() noexcept {
                    if (tri != triend) {
                        for (; diagmin != diagend and diagmin->first < tri->first.first; ++diagmin) ;
                        assert(diagmin != diagend && diagmin->first == tri->first.first);
                        for (diagmax = diagmin; diagmax != diagend and diagmax->first < tri->first.second; ++diagmax) ;
                        assert(diagmax != diagend && diagmax->first == tri->first.second);
                    } else {
                        diagmin = diagend;
                        diagmax = diagend;
                    }
                }
                
            public:            
                ConstTriIt(
                    const typename std::map<Id, Diag>::const_iterator &diag_begin,
                    const typename std::map<Id, Diag>::const_iterator &diag_end,
                    const typename std::map<IdPair, Real>::const_iterator &tri_begin,
                    const typename std::map<IdPair, Real>::const_iterator &tri_end,
                    const bool skipzero
                ) noexcept:
                diagend {diag_end}, triend {tri_end}, skipmissing {skipzero},
                diagmin {diag_begin}, diagmax {diag_begin}, tri {tri_begin} {
                    if (skipmissing) {
                        this->update_diag();
                    } else if (diagmax != diagend && ++diagmax == diagend) {
                        diagmin = diagend;
                    }
                }
                
                inline ConstTriIt(
                    const typename std::map<Id, Diag>::const_iterator &diag_end,
                    const typename std::map<IdPair, Real>::const_iterator &tri_end
                ) noexcept:
                diagend {diag_end}, triend {tri_end},
                diagmin {diag_end}, diagmax {diag_end}, tri {tri_end} {
                    ;
                }
                
                void operator++() noexcept {
                    if (skipmissing) {
                        ++tri;
                        this->update_diag();
                    } else {
                        ++diagmax;
                        if (diagmax == diagend) {
                            ++diagmin;
                            diagmax = diagmin;
                            if (diagmax != diagend && ++diagmax != diagend) {
                                const IdPair id {diagmin->first, diagmax->first};
                                for (; tri != triend && tri->first < id; ++tri) ;
                            } else {
                                diagmin = diagend;
                            }
                        }
                    }
                }
                
                const Real &operator*() const noexcept {
                    if (skipmissing) {
                        assert(tri != triend);
                        return tri->second;
                    } else {
                        assert(diagmin != diagend and diagmax != diagend);
                        const IdPair id {diagmin->first, diagmax->first};
                        if (tri != triend && tri->first == id) {
                            return tri->second;
                        } else {
                            return zero;
                        }
                    }
                }
                
                inline Id id1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->first;
                }
                
                inline Id id2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->first;
                }
                
                inline IdPair id() const noexcept {
                    assert(this->id1() < this->id2());
                    return {this->id1(), this->id2()};
                }
                
                inline const Diag &diag1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->second;
                }
                
                inline const Diag &diag2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->second;
                }
                
                inline friend bool operator!=(const ConstTriIt &it1, const ConstTriIt &it2) noexcept {
                    assert(it1.diagend == it2.diagend and it1.triend == it2.triend);
                    return it1.diagmin != it2.diagmin and it1.diagmax != it2.diagmax;
                }
            };
            
            ConstTriIt ctbegin(const bool skipzero) const noexcept {
                return ConstTriIt(diagmap.cbegin(), diagmap.cend(), trimap.cbegin(), trimap.cend(), skipzero);
            }
        
            inline ConstTriIt ctend() const noexcept {
                return ConstTriIt(diagmap.cend(), trimap.cend());
            }
            
            class ConstHessIt {
            private:
                typename std::map<Id, Diag>::const_iterator diagmin;
                typename std::map<Id, Diag>::const_iterator diagmax;
                const typename std::map<Id, Diag>::const_iterator diagend;
                typename std::map<IdPair, Real>::const_iterator tri;
                const typename std::map<IdPair, Real>::const_iterator triend;
                const bool skipmissing = true;
                
                void update_diag() {
                    if (tri != triend) {
                        for (; diagmin != diagend and diagmin->first < tri->first.first; ++diagmin) ;
                        assert(diagmin != diagend && diagmin->first == tri->first.first);
                        for (diagmax = diagmin; diagmax != diagend and diagmax->first < tri->first.second; ++diagmax) ;
                        assert(diagmax != diagend && diagmax->first == tri->first.second);
                    } else {
                        diagmin = diagend;
                        diagmax = diagend;
                    }
                }
                    
            public:            
                ConstHessIt(
                    const typename std::map<Id, Diag>::const_iterator &diag_begin,
                    const typename std::map<Id, Diag>::const_iterator &diag_end,
                    const typename std::map<IdPair, Real>::const_iterator &tri_begin,
                    const typename std::map<IdPair, Real>::const_iterator &tri_end,
                    const bool skipzero
                ) noexcept:
                diagend {diag_end}, triend {tri_end}, skipmissing {skipzero},
                diagmin {diag_begin}, diagmax {diag_begin}, tri {tri_begin} {
                    ;
                }
                
                inline ConstHessIt(
                    const typename std::map<Id, Diag>::const_iterator &diag_end,
                    const typename std::map<IdPair, Real>::const_iterator &tri_end
                ) noexcept:
                diagend {diag_end}, triend {tri_end},
                diagmin {diag_end}, diagmax {diag_end}, tri {tri_end} {
                    ;
                }
                
                void operator++() noexcept {
                    assert(diagmin != diagend and diagmax != diagend);
                    if (diagmin == diagmax) {
                        ++diagmax;
                    } else {
                        ++tri;
                    }
                    assert(diagmax != diagend or tri == triend);
                    assert((tri == triend || tri->first >= IdPair {diagmin->first, diagmax->first}));
                    if (tri == triend || tri->first.first > diagmin->first) {
                        ++diagmin;
                        diagmax = diagmin;
                    } else if (skipmissing) {
                        for (; diagmax != diagend && diagmax->first < tri->first.second; ++diagmax) ;
                        assert(diagmax != diagend && diagmax->first == tri->first.second);
                    }
                    assert(diagmax != diagend or diagmin == diagend);
                }
                
                const Real &operator*() const noexcept {
                    assert(diagmin != diagend and diagmax != diagend);
                    assert(diagmin == diagmax or not skipmissing or tri != triend);
                    if (diagmin == diagmax) {
                        return diagmin->second.hhess;
                    } else if (skipmissing) {
                        return tri->second;
                    } else if (tri != triend && tri->first == IdPair {diagmin->first, diagmax->first}) {
                        return tri->second;
                    } else {
                        return zero;
                    }
                }
                
                inline Id id1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->first;
                }
                
                inline Id id2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->first;
                }
                
                inline IdPair id() const noexcept {
                    assert(this->id1() < this->id2());
                    return {this->id1(), this->id2()};
                }
                
                inline const Diag &diag1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->second;
                }
                
                inline const Diag &diag2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->second;
                }
                
                inline friend bool operator!=(const ConstHessIt &it1, const ConstHessIt &it2) noexcept {
                    assert(it1.diagend == it2.diagend and it1.triend == it2.triend);
                    return it1.diagmin != it2.diagmin and it1.diagmax != it2.diagmax;
                }
            };

            ConstHessIt chbegin(const bool skipzero) const noexcept {
                return ConstHessIt(diagmap.cbegin(), diagmap.cend(), trimap.cbegin(), trimap.cend(), skipzero);
            }
        
            inline ConstHessIt chend() const noexcept {
                return ConstHessIt(diagmap.cend(), trimap.cend());
            }
            
            template<typename OtherReal>
            friend class HessGrad;
            
            template<typename OtherReal>
            operator HessGrad<OtherReal>() const {
                HessGrad<OtherReal> hg;
                for (const auto &it : diagmap) {
                    typename HessGrad<OtherReal>::Diag &node = hg.diagmap[it.first];
                    const Diag &diag = it.second;
                    node.grad = diag.grad;
                    node.hhess = diag.hhess;
                    std::copy(diag.mom.begin(), diag.mom.end(), node.mom.begin());
                }
                for (const auto &it : trimap) {
                    hg.trimap[it.first] = it.second;
                }
                return hg;
            }
        };
        
        template<typename Real>
        const Real HessGrad<Real>::zero {0};
    
        template<typename Real>
        const typename HessGrad<Real>::Diag HessGrad<Real>::zerodiag;
    }
}

#endif /* end of include guard: UNCERTAINTIES_HESSGRAD_HPP_84C05D5B */

