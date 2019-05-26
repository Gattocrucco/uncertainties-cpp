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
        using M7P = std::shared_ptr<std::array<Real, 7>>;
    
        template<int n, typename Real>
        inline const Real &v(const M7P<Real> &p) noexcept {
            static_assert(n >= 2 and n <= 8, "it must be 2 <= n <= 8");
            return std::get<n - 2>(*p);
        }
    
        template<typename Real>
        class HessGrad {
        public:
            struct Diag {
                Real grad {0};
                Real hhess {0}; // half second derivative
                M7P<Real> mom;
            private:
                std::map<Id, Real> row;
                
                template<typename>
                friend class HessGrad;
            };
        
        private:
            std::map<Id, Diag> nodes;
        
        public:
            inline typename std::map<Id, Diag>::size_type size() const noexcept {
                return this->nodes.size();
            }
        
            inline const Diag &diag(const Id id) const {
                return this->nodes.at(id);
            }
        
            inline Diag &diag(const Id id) {
                return this->nodes[id];
            }
            
            Diag diag_get(const Id id) const {
                using It = typename std::map<Id, Diag>::const_iterator;
                const It it = this->nodes.find(id);
                if (it != this->nodes.end()) {
                    return it->second;
                } else {
                    return Diag();
                }
            }
            
            const Real &tri(const Id minid, const Id maxid) const {
                assert(minid < maxid);
                auto n = this->nodes.find(minid);
                assert(n != this->nodes.end());
                assert(this->nodes.find(maxid) != this->nodes.end());
                return n->second.row.at(maxid);
            }

            Real &tri(const Id minid, const Id maxid) {
                assert(minid < maxid);
                return this->nodes[minid].row[maxid];
            }

            Real tri_get(const Id minid, const Id maxid) const {
                assert(minid < maxid);
                auto n = this->nodes.find(minid);
                if (n == this->nodes.end()) {
                    return 0;
                }
                auto t = n->second.row.find(maxid);
                if (t == n->second.row.end()) {
                    return 0;
                }
                assert(this->nodes.find(maxid) != this->nodes.end());
                return t->second;
            }
            
            inline Real &hhess(const Id minid, const Id maxid) {
                return minid != maxid ? this->tri(minid, maxid) : this->diag(minid).hhess;
            }
            
            inline Real hhess_get(const Id id1, const Id id2) const {
                return id1 != id2 ? this->tri_get(std::min(id1, id2), std::max(id1, id2)) : this->diag_get(id1).hhess;
            }
            
            using DiagIt = typename std::map<Id, Diag>::iterator;
        
            inline DiagIt dbegin() {
                return nodes.begin();
            }
        
            inline DiagIt dend() {
                return nodes.end();
            }
            
            using ConstDiagIt = typename std::map<Id, Diag>::const_iterator;

            inline ConstDiagIt cdbegin() const {
                return nodes.cbegin();
            }
        
            inline ConstDiagIt cdend() const {
                return nodes.cend();
            }
            
            class TriIt {
            private:
                typename std::map<Id, Diag>::iterator itnmin;
                typename std::map<Id, Diag>::iterator itnmax;
                const typename std::map<Id, Diag>::iterator itnend;
                typename std::map<Id, Real>::iterator ithhess;
        
                void update_itnmax(const Id id) noexcept {
                    for (; this->itnmax->first != id; ++this->itnmax) {
                        assert(this->itnmax != this->itnend);
                    }
                }
            
                void update() {
                    this->itnmax = this->itnmin;
                    if (this->itnmin != this->itnend) {
                        this->ithhess = this->itnmin->second.row.begin();
                        if (this->ithhess != this->itnmin->second.row.end()) {
                            this->update_itnmax(this->ithhess->first);
                        }
                    }
                }
            
            public:            
                TriIt(const typename std::map<Id, Diag>::iterator &nodes_begin,
                      const typename std::map<Id, Diag>::iterator &nodes_end):
                itnend {nodes_end} {
                    for (this->itnmin = nodes_begin; this->itnmin != nodes_end; ++(this->itnmin)) {
                        if (this->itnmin->second.row.size() > 0) {
                            break;
                        }
                    }
                    this->update();
                }
                
                inline explicit TriIt(
                    const typename std::map<Id, Diag>::iterator &nodes_end
                ): itnend {nodes_end}, itnmin {nodes_end}, itnmax {nodes_end} {
                    ;
                }
                
                void operator++() {
                    assert(this->itnmin != this->itnend);
                    assert(this->ithhess != this->itnmin->second.row.end());
                    ++this->ithhess;
                    if (this->ithhess == this->itnmin->second.row.end()) {
                        ++this->itnmin;
                        this->update();
                    }
                }
                
                inline Real &operator*() const noexcept {
                    assert(this->itnmin != this->itnend);
                    assert(this->ithhess != this->itnmin->second.row.end());
                    return this->ithhess->second;
                }
                
                inline Real *operator->() const noexcept {
                    return &(this->operator*());
                }
                
                inline Id id1() const noexcept {
                    assert(this->itnmin != this->itnend);
                    return this->itnmin->first;
                }
                
                inline Id id2() const noexcept {
                    assert(this->itnmax != this->itnend);
                    return this->itnmax->first;
                }
                
                inline const Diag &diag1() const noexcept {
                    assert(this->itnmin != this->itnend);
                    return this->itnmin->second;
                }
                
                inline const Diag &diag2() const noexcept {
                    assert(this->itnmax != this->itnend);
                    return this->itnmax->second;
                }
                
                // IMPORTANT:
                // operator!= is for iterating
                // operator<= and operator== are for comparing ids of iterators
                // from different objects
                                
                inline friend bool operator!=(const TriIt &it1, const TriIt &it2) noexcept {
                    if (it1.itnmin != it2.itnmin and it1.itnmax != it2.itnmax) {
                        assert(it1.ithhess != it2.ithhess);
                        return true;
                    }
                    // do not check because ithhess can not be set to nullptr
                    // so the end interator has a random ithhess
                    // assert(it1.ithhess != it2.ithhess)
                    return false;
                }
                
                inline friend bool operator<=(const TriIt &it1, const TriIt &it2) noexcept {
                    return it1.id1() < it2.id1() || (it1.id1() == it2.id1() && it1.id2() <= it2.id2());
                }
                
                inline friend bool operator==(const TriIt &it1, const TriIt &it2) noexcept {
                    return it1.id1() == it2.id1() and it1.id2() == it2.id2();
                }
            };
        
            TriIt tbegin() {
                return TriIt(nodes.begin(), nodes.end());
            }
        
            inline TriIt tend() {
                return TriIt(nodes.end());
            }

            class ConstTriIt {
            private:
                typename std::map<Id, Diag>::const_iterator itnmin;
                typename std::map<Id, Diag>::const_iterator itnmax;
                const typename std::map<Id, Diag>::const_iterator itnend;
                typename std::map<Id, Real>::const_iterator ithhess;
                bool skip;
                    
                void update_itnmax_skip(const Id id) noexcept {
                    for (; this->itnmax->first != id; ++this->itnmax) {
                        assert(this->itnmax != this->itnend);
                    }
                }
            
                void update_skip() {
                    this->itnmax = this->itnmin;
                    if (this->itnmin != this->itnend) {
                        this->ithhess = this->itnmin->second.row.begin();
                        if (this->ithhess != this->itnmin->second.row.end()) {
                            this->update_itnmax_skip(this->ithhess->first);
                        }
                    }
                }
            
            public:            
                ConstTriIt(
                    const typename std::map<Id, Diag>::const_iterator &nodes_begin,
                    const typename std::map<Id, Diag>::const_iterator &nodes_end,
                    const bool skipzero
                ): itnend {nodes_end}, skip {skipzero} {
                    if (this->skip) {
                        for (this->itnmin = nodes_begin; this->itnmin != nodes_end; ++(this->itnmin)) {
                            if (this->itnmin->second.row.size() > 0) {
                                break;
                            }
                        }
                        this->update_skip();
                    } else {
                        this->itnmin = nodes_begin;
                        this->itnmax = this->itnmin;
                        if (this->itnmax != this->itnend) {
                            ++(this->itnmax);
                            this->ithhess = this->itnmin->second.row.begin();
                        }
                    }
                }
                
                inline explicit ConstTriIt(
                    const typename std::map<Id, Diag>::const_iterator &nodes_end
                ): itnend {nodes_end}, itnmin {nodes_end}, itnmax {nodes_end} {
                    ;
                }
                
                void operator++() {
                    if (this->skip) {
                        assert(this->itnmin != this->itnend);
                        assert(this->ithhess != this->itnmin->second.row.end());
                        ++this->ithhess;
                        if (this->ithhess == this->itnmin->second.row.end()) {
                            ++this->itnmin;
                            this->update_skip();
                        }
                    } else {
                        assert(this->itnmax != this->itnend);
                        ++(this->itnmax);
                        if (this->itnmax != this->itnend) {
                            for (; this->ithhess != this->itnmin->second.row.end(); ++(this->ithhess)) {
                                if (this->ithhess->first >= this->itnmax->first) {
                                    break;
                                }
                            }
                        } else {
                            ++(this->itnmin);
                            this->itnmax = this->itnmin;
                            if (this->itnmax != this->itnend) {
                                ++(this->itnmax);
                                this->ithhess = this->itnmin->second.row.begin();   
                            }
                        }
                    }
                }
                
                Real operator*() const noexcept {
                    if (this->skip) {
                        assert(this->itnmin != this->itnend);
                        assert(this->ithhess != this->itnmin->second.row.end());
                        return this->ithhess->second;
                    }
                    assert(this->itnmax != this->itnend);
                    if (this->ithhess != this->itnmin->second.row.end() && this->ithhess->first == this->itnmax->first) {
                        return this->ithhess->second;
                    } else {
                        return 0;
                    }
                }
                
                inline Id id1() const noexcept {
                    assert(this->itnmin != this->itnend);
                    return this->itnmin->first;
                }
                
                inline Id id2() const noexcept {
                    assert(this->itnmax != this->itnend);
                    return this->itnmax->first;
                }
                
                inline std::pair<Id, Id> id() const noexcept {
                    assert(this->id1() < this->id2());
                    return {this->id1(), this->id2()};
                }
                
                inline const Diag &diag1() const noexcept {
                    assert(this->itnmin != this->itnend);
                    return this->itnmin->second;
                }
                
                inline const Diag &diag2() const noexcept {
                    assert(this->itnmax != this->itnend);
                    return this->itnmax->second;
                }
                
                inline friend bool operator!=(const ConstTriIt &it1, const ConstTriIt &it2) noexcept {
                    return it1.itnmin != it2.itnmin and it1.itnmax != it2.itnmax;
                }

                inline friend bool operator<=(const ConstTriIt &it1, const ConstTriIt &it2) noexcept {
                    return it1.id1() < it2.id1() || (it1.id1() == it2.id1() && it1.id2() <= it2.id2());
                }
                
                inline friend bool operator==(const ConstTriIt &it1, const ConstTriIt &it2) noexcept {
                    return it1.id1() == it2.id1() and it1.id2() == it2.id2();
                }
            };
            
            ConstTriIt ctbegin(const bool skipzero) const {
                return ConstTriIt(nodes.cbegin(), nodes.cend(), skipzero);
            }
        
            inline ConstTriIt ctend() const {
                return ConstTriIt(nodes.cend());
            }
            
            class ConstHessIt {
            private:
                const typename std::map<Id, Diag>::const_iterator end;
                typename std::map<Id, Diag>::const_iterator itmin;
                typename std::map<Id, Diag>::const_iterator itmax;
                typename std::map<Id, Real>::const_iterator ittri;
                bool ondiag = true;
                const bool skip = true;
                
            public:
                ConstHessIt(const typename std::map<Id, Diag>::const_iterator &nodes_begin,
                            const typename std::map<Id, Diag>::const_iterator &nodes_end,
                            const bool skipzero):
                end {nodes_end}, itmin {nodes_begin}, itmax {nodes_begin},
                skip {skipzero} {
                    ;
                }
                
                ConstHessIt(const typename std::map<Id, Diag>::const_iterator &nodes_end):
                end {nodes_end}, itmin {nodes_end}, itmax {nodes_end} {
                    ;
                }
                
                void operator++() {
                    assert(itmin != end);
                    assert(itmax != end);
                    if (skip) {
                        if (ondiag) {
                            ittri = itmin->second.row.cbegin();
                            if (ittri != itmin->second.row.cend()) {
                                ondiag = false;
                            } else {
                                ++itmin;
                                itmax = itmin;
                            }
                        } else {
                            ++ittri;
                            if (ittri == itmin->second.row.cend()) {
                                ++itmin;
                                itmax = itmin;
                                ondiag = true;
                            }
                        }
                        if (not ondiag) {
                            for (; itmax->first < ittri->first; ++itmax) ;
                            assert(itmax->first == ittri->first);
                        }
                    } else {
                        if (ondiag) {
                            ittri = itmin->second.row.cbegin();
                            ++itmax;
                            ondiag = false;
                        } else {
                            ++itmax;
                            if (itmax == end) {
                                ++itmin;
                                itmax = itmin;
                                ondiag = true;
                            } else {
                                const auto triend = itmin->second.row.cend();
                                while (ittri != triend && ittri->first < itmax->first) {
                                    ++ittri;
                                }
                            }
                        }
                    }
                }
                
                Real operator*() const noexcept {
                    assert(itmin != end);
                    assert(itmax != end);
                    if (ondiag) {
                        assert(itmax == itmin);
                        return itmin->second.hhess;
                    } else {
                        assert(itmin->first <= itmax->first);
                        const auto triend = itmin->second.row.cend();
                        assert(ittri == triend || ittri->first >= itmax->first);
                        if (ittri != triend && ittri->first == itmax->first) {
                            return ittri->second;
                        } else {
                            return 0;
                        }
                    }
                }
                
                inline Id id1() const noexcept {
                    assert(itmin != end);
                    return itmin->first;
                }
                
                inline Id id2() const noexcept {
                    assert(itmax != end);
                    return itmax->first;
                }
                
                inline const Diag &diag1() const noexcept {
                    assert(itmin != end);
                    return itmin->second;
                }
                
                inline const Diag &diag2() const noexcept {
                    assert(itmax != end);
                    return itmax->second;
                }
                
                inline friend bool operator!=(const ConstHessIt &it1, const ConstHessIt &it2) noexcept {
                    assert(it1.end == it2.end);
                    return it1.itmin != it2.itmin and it1.itmax != it2.itmax;
                }
            };
            
            ConstHessIt chbegin(const bool skipzero) const {
                return ConstHessIt(this->nodes.begin(), this->nodes.end(), skipzero);
            }
            
            ConstHessIt chend() const {
                return ConstHessIt(this->nodes.end());
            }

            template<typename OtherReal>
            friend class HessGrad;
            
            template<typename OtherReal>
            operator HessGrad<OtherReal>() const {
                HessGrad<OtherReal> hg;
                for (const auto &it : this->nodes) {
                    typename HessGrad<OtherReal>::Diag &node = hg.nodes[it.first];
                    const Diag &diag = it.second;
                    node.grad = diag.grad;
                    node.hhess = diag.hhess;
                    node.mom = M7P<OtherReal>(new std::array<OtherReal, 7>);
                    std::copy(diag.mom->begin(), diag.mom->end(), node.mom->begin());
                    for (const auto &it2 : it.second.row) {
                        node.row[it2.first] = it2.second;
                    }
                }
                return hg;
            }
        };
    }
}

#endif /* end of include guard: UNCERTAINTIES_HESSGRAD_HPP_84C05D5B */

