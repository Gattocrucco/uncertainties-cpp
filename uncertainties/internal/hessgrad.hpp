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
        
        // Interface to store the moments of elementary variables. It is
        // currently just a wrapper around a std::array. I made it into a class
        // so that in the future it may become a smart pointer, so that arrays
        // of moments are shared between different dipendent variables without
        // copying them over. Since it is only 6 numbers I guess it is better
        // to copy them. It could be specialized to smart pointer only when Real
        // is something big.
        template<typename Real>
        class Moments {
        private:
            std::array<Real, 6> m; // standardized moments 3 to 8
        
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
    
        // Get the nth standardized moment out of a Moments objects. May need
        // to drop the noexcept if Moments becomes a smart pointer.
        template<int n, typename Real>
        inline const Real &v(const Moments<Real> &p) noexcept {
            static_assert(n >= 3 and n <= 8, "it must be 3 <= n <= 8");
            return p.template get<n - 3>();
        }
    
        // Class that implements storing variable dependency for UReal2. It
        // stores in a sparse format the hessian and gradient w.r.t.
        // elementary variables, along with their moments. Remember that
        // elementary variables are standardized, i.e. E[x] = 0, E[x^2] = 1,
        // so we just store standardized moments from 3 onward. This class is
        // needed to separate implementation of matrix storage from UReal2
        // functionality. Thus, the class provides many specific access patterns
        // instead of minimal random access so that it can take care of
        // choosing the optimal implementation. We actually always store *half*
        // the second derivatives because in Taylor expansions they would
        // always get a factor 1/2.
        template<typename Real>
        class HessGrad {
        public:
            struct Diag {
                Real grad {0}; // first derivative
                Real hhess {0}; // half second derivative
                Moments<Real> mom; // standardized moments 3 to 8
            };
            
            using IdPair = std::pair<Id, Id>;
        
        private:
            // map containing gradient, diagonal of the hessian, and moments
            std::map<Id, Diag> diagmap;
            
            // map containing out of diagonal elements of half the hessian
            std::map<IdPair, Real> trimap;
            
            // The maps are split in this way (instead of, say, a map for
            // gradient and moments and one for the hessian) because often it
            // is just necessary to iterate over gradient, hessian diagonal and
            // moments, without looking at out of diagonal hessian.
            
            static const Real zero;
            static const Diag zerodiag;
            // These two variables get their initialization out of the class
            // definition because yes the compiler said to do so.
        
        public:
            // Return number of variables stored. We just need to check the
            // size of the diagmap because, since it contains the moments, there
            // must be an entry even if gradient and hessian diagonal are zero
            // for a given variable.
            inline typename std::map<Id, Diag>::size_type size() const noexcept {
                return diagmap.size();
            }
        
            // Return diag object (gradient, hessian diagonal, moments) for
            // specific variable.
            inline const Diag &diag(const Id id) const {
                return diagmap.at(id);
            }
        
            // Mutable version of the above; in case of missing variable, the
            // variable is added.
            inline Diag &diag(const Id id) {
                return diagmap[id];
            }
            
            // Return diag object, but instead of throwing an exception if the
            // variable is not in the map, it returns a diagonal with zero
            // entries. The gradient and half hessian are zero, but the moments
            // may not since we assume that they would be multiplied by the
            // derivatives anyway.
            const Diag &diag_get(const Id id) const noexcept {
                using It = typename std::map<Id, Diag>::const_iterator;
                const It it = diagmap.find(id);
                if (it != diagmap.end()) {
                    return it->second;
                } else {
                    return zerodiag;
                }
            }
            
            // Return out of diagonal half hessian element. The double index
            // must be sorted. Exception if missing from map.
            const Real &tri(const Id minid, const Id maxid) const {
                assert(minid < maxid);
                return trimap.at({minid, maxid});
            }

            // Mutable version of the above, so in case of missing variable,
            // it is inserted anew.
            Real &tri(const Id minid, const Id maxid) {
                assert(minid < maxid);
                return trimap[{minid, maxid}];
            }

            // Like above, but for missing variable zero is returned instead
            // of adding the variable or throwing an exception.
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
            
            // Return crude pointer to out of diagonal half hessian. The IDs
            // can be in any order. If the variable is missing, returns NULL.
            const Real *tri_find(const Id id1, const Id id2) const noexcept {
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                using It = typename std::map<IdPair, Real>::const_iterator;
                const It it = trimap.find({minid, maxid});
                if (it != trimap.end()) {
                    return &it->second;
                } else {
                    return nullptr;
                }
            }
            
            // Return reference to half hessian element, either on or out of
            // diagonal. IDs must be sorted. If the variable is missing, it is
            // created.
            // **DANGER**: you can create an out of diagonal element without a
            // corresponding diagonal element in this way. This breaks
            // assumptions.
            inline Real &hhess(const Id minid, const Id maxid) {
                return minid != maxid ? this->tri(minid, maxid) : this->diag(minid).hhess;
            }
            
            // Return half hessian element. IDs need not be sorted. If the
            // element is missing, zero is returned.
            inline const Real &hhess_get(const Id id1, const Id id2) const noexcept {
                return id1 != id2 ? this->tri_get(std::min(id1, id2), std::max(id1, id2)) : this->diag_get(id1).hhess;
            }
            
            // Iterator to Diag objects. Actually very simple now since the
            // implementation is an actual collection of Diags.
            using DiagIt = typename std::map<Id, Diag>::iterator;
            
            // IMPORTANT: I don't remember if the code in UReal2 relies on the
            // specific ordering of these iterators. Check that.
        
            // Begin iterator the Diag objects.
            inline DiagIt dbegin() noexcept {
                return diagmap.begin();
            }
            
            // End iterator to the Diag objects.
            inline DiagIt dend() noexcept {
                return diagmap.end();
            }
            
            // Const version of the above.
            using ConstDiagIt = typename std::map<Id, Diag>::const_iterator;

            inline ConstDiagIt cdbegin() const noexcept {
                return diagmap.cbegin();
            }
        
            inline ConstDiagIt cdend() const noexcept {
                return diagmap.cend();
            }
            
            // Class to iterate on out of diagonal half hessian elements without
            // modifying them, while keeping track of corresponding diagonal
            // elements.
            class ConstTriIt {
            private:
                typename std::map<Id, Diag>::const_iterator diagmin;
                typename std::map<Id, Diag>::const_iterator diagmax;
                const typename std::map<Id, Diag>::const_iterator diagend;
                typename std::map<IdPair, Real>::const_iterator tri;
                const typename std::map<IdPair, Real>::const_iterator triend;
                const bool skipmissing = true;
                
                // This function make iterators to the diagonal "follow" the
                // iterator to out of the diagonal. It is used if
                // skipmissing=true. Otherwise, it is the out of diagonal that
                // follows the diagonal.
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
                
                // Construct the begin iterator given begin/end iterators to the
                // diagmap and the trimap. skipzero=true means that missing
                // out of diagonal elements will not be iterated over (recall
                // we are sparse), skipzero=false means they will be iterated
                // over and provided as zero through the interface.
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
                
                // Construct the end iterator given end iterators to diagmap
                // and trimap.
                inline ConstTriIt(
                    const typename std::map<Id, Diag>::const_iterator &diag_end,
                    const typename std::map<IdPair, Real>::const_iterator &tri_end
                ) noexcept:
                diagend {diag_end}, triend {tri_end},
                diagmin {diag_end}, diagmax {diag_end}, tri {tri_end} {
                    ;
                }
                
                // Advance the iterator.
                void operator++() noexcept {
                    if (skipmissing) {
                        ++tri;
                        this->update_diag();
                    } else {
                        ++diagmax;
                        if (diagmax == diagend) {
                            ++diagmin;
                            diagmax = diagmin;
                            if (diagmax != diagend) {
                                ++diagmax;
                            }
                        }
                        if (diagmax != diagend) {
                            const IdPair id {diagmin->first, diagmax->first};
                            for (; tri != triend && tri->first < id; ++tri) ;
                        } else {
                            diagmin = diagend;
                        }
                    }
                }
                
                // Access the iterator, returning the out of diagonal half
                // hessian.
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
                
                // "Row" ID.
                inline Id id1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->first;
                }
                
                // "Column" ID.
                inline Id id2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->first;
                }
                
                // ID pair.
                inline IdPair id() const noexcept {
                    assert(this->id1() < this->id2());
                    return {this->id1(), this->id2()};
                }
                
                // Diag object corresponing to current row.
                inline const Diag &diag1() const noexcept {
                    assert(diagmin != diagend);
                    return diagmin->second;
                }
                
                // Diag object corresponding to current column.
                inline const Diag &diag2() const noexcept {
                    assert(diagmax != diagend);
                    return diagmax->second;
                }
                
                // Iterator comparison.
                inline friend bool operator!=(const ConstTriIt &it1, const ConstTriIt &it2) noexcept {
                    assert(it1.diagend == it2.diagend and it1.triend == it2.triend);
                    return it1.diagmin != it2.diagmin and it1.diagmax != it2.diagmax;
                }
            };
            
            // Begin iterator to out of diagonal half hessian.
            ConstTriIt ctbegin(const bool skipzero) const noexcept {
                return ConstTriIt(diagmap.cbegin(), diagmap.cend(), trimap.cbegin(), trimap.cend(), skipzero);
            }
        
            // End iterator to out of diagonal half hessian.
            inline ConstTriIt ctend() const noexcept {
                return ConstTriIt(diagmap.cend(), trimap.cend());
            }
            
            // Iterator to the full hessian. Works similarly to ConstTriIt but
            // also iterate over diagonal elements. In this case skipmissing
            // still means that missing out of diagonal entries will be skipped.
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
            
            // Conversion to HessGrad with different numerical type. Used
            // to implement the analogous conversion of UReal2.
            template<typename OtherReal>
            operator HessGrad<OtherReal>() const {
                HessGrad<OtherReal> hg;
                for (const auto &it : diagmap) {
                    typename HessGrad<OtherReal>::Diag &node = hg.diagmap[it.first];
                    const Diag &diag = it.second;
                    node.grad = diag.grad;
                    node.hhess = diag.hhess;
                    std::copy(diag.mom.begin(), diag.mom.end(), node.mom.begin());
                    // This copy bit remains the same even if Moments becomes
                    // a smart pointer, because we are changing type.
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

