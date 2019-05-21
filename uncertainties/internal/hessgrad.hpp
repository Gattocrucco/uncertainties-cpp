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
                Real grad;
                Real hhess; // half second derivative
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
        
            inline const Diag &diag(const Id id) const noexcept {
                return this->nodes.at(id);
            }
        
            inline Diag &diag(const Id id) {
                return this->nodes[id];
            }

            const Real &tri(const Id id1, const Id id2) const {
                if (id1 == id2) {
                    throw std::runtime_error("tri: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                auto n = this->nodes.find(minid);
                if (n == this->nodes.end()) {
                    throw std::runtime_error("tri_set: !n");
                }
                auto N = this->nodes.find(maxid);
                if (N == this->nodes.end()) {
                    throw std::runtime_error("tri_set: !N");
                }
                return n->second.row.at(maxid);
            }

            Real &tri(const Id id1, const Id id2) {
                if (id1 == id2) {
                    throw std::runtime_error("tri: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                auto n = this->nodes.find(minid);
                if (n == this->nodes.end()) {
                    throw std::runtime_error("tri_set: !n");
                }
                auto N = this->nodes.find(maxid);
                if (N == this->nodes.end()) {
                    throw std::runtime_error("tri_set: !N");
                }
                return n->second.row[maxid];
            }

            const Real &tri_get(const Id id1, const Id id2) const {
                if (id1 == id2) {
                    throw std::runtime_error("tri_get: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                auto n = this->nodes.find(minid);
                if (n == this->nodes.end()) {
                    return 0;
                }
                auto t = n->second.row.find(maxid);
                if (t == n->second.row.end()) {
                    return 0;
                }
                auto N = this->nodes.find(maxid);
                if (N == this->nodes.end()) {
                    throw std::runtime_error("tri_get: !N");
                }
                return *t;
            }
        
            class DiagIt {
            private:
                typename std::map<Id, Diag>::iterator it;
            
            public:                
                DiagIt(const typename std::map<Id, Diag>::iterator &i): it {i} {
                    ;
                }
                inline void operator++() noexcept {
                    ++it;
                }
                inline std::pair<Id, Diag> &operator*() const noexcept {
                    return *it;
                }
                friend inline bool operator!=(const DiagIt &it1, const DiagIt &it2) {
                    return it1.it != it2.it;
                }
            };

            DiagIt dbegin() {
                return DiagIt(nodes.begin());
            }
        
            DiagIt dend() {
                return DiagIt(nodes.end());
            }

            class ConstDiagIt {
            private:
                typename std::map<Id, Diag>::const_iterator it;
            
            public:
                ConstDiagIt(const typename std::map<Id, Diag>::const_iterator &i): it {i} {
                    ;
                }
                inline void operator++() noexcept {
                    ++it;
                }
                inline decltype(*it) operator*() const noexcept {
                    return *it;
                }
                inline decltype(it) operator->() const noexcept {
                    return it;
                }
                friend inline bool operator!=(const ConstDiagIt &it1, const ConstDiagIt &it2) {
                    return it1.it != it2.it;
                }
            };
        
            ConstDiagIt cdbegin() const {
                return ConstDiagIt(nodes.cbegin());
            }
        
            ConstDiagIt cdend() const {
                return ConstDiagIt(nodes.cend());
            }

            template<typename DiagIterator, typename RealIterator>
            class TriItTempl {
            private:
                DiagIterator itnmin;
                DiagIterator itnmax;
                const DiagIterator itnend;
                RealIterator ithhess;
        
                void update_itnmax(const Id id) {
                    for (; this->itnmax->first != id; ++this->itnmax) {
                        if (this->itnmax == this->itnend) {
                            throw std::runtime_error("TriIt::update_itnmax: itnmax == itnend");
                        }
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
                TriItTempl(const DiagIterator &nodes_begin,
                           const DiagIterator &nodes_end):
                itnend {nodes_end} {
                    for (this->itnmin = nodes_begin; this->itnmin != nodes_end; ++(this->itnmin)) {
                        if (this->itnmin->second.row.size() > 0) {
                            break;
                        }
                    }
                    this->update();
                }
                void operator++() {
                    if (this->itnmin != this->itnend) {
                        if (this->ithhess == this->itnmin->second.row.end()) {
                            throw std::runtime_error("TriIt::operator++: ithhess == end");
                        }
                        ++this->ithhess;
                        if (this->ithhess == this->itnmin->second.row.end()) {
                            ++this->itnmin;
                            this->update();
                        }
                    }
                }
                decltype(ithhess->second) operator*() const {
                    if (this->ithhess == this->itnmin->second.row.end()) {
                        throw std::runtime_error("TriIt::operator*: ithhess == end");
                    }
                    return this->ithhess->second;
                }
                Id id1() const {
                    if (this->itnmin == this->itnend) {
                        throw std::runtime_error("TriIt::id1: itnmin == end");
                    }
                    return this->itnmin->first;
                }
                Id id2() const {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("TriIt::id2 itnmax == end");
                    }
                    return this->itnmax->first;
                }
                const Diag &diag1() const {
                    if (this->itnmin == this->itnend) {
                        throw std::runtime_error("TriIt::diag1: itnmin == end");
                    }
                    return this->itnmin->second;
                }
                const Diag &diag2() const {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("TriIt::diag2: itnmax == end");
                    }
                    return this->itnmax->second;
                }
                using ThisType = TriItTempl<DiagIterator, RealIterator>;
                friend bool operator!=(const ThisType &it1, const ThisType &it2) {
                    if (it1.itnmin != it2.itnmin and it1.itnmax != it2.itnmax) {
                        if (it1.ithhess == it2.ithhess) {
                            throw std::runtime_error("TriIt::operator!=: ithhess1 == ithhess2");
                        }
                        return true;
                    }
                    // do not check because ithhess can not be set to nullptr
                    // so the end interator has a random ithhess
                    // if (it1.ithhess != it2.ithhess) {
                    //     throw std::runtime_error("TriIt::operator!=: ithhess1 != ithhess2");
                    // }
                    return false;
                }
            };
        
            using TriIt = TriItTempl<typename std::map<Id, Diag>::iterator, typename std::map<Id, Real>::iterator>;
            using ConstTriIt = TriItTempl<typename std::map<Id, Diag>::const_iterator, typename std::map<Id, Real>::const_iterator>;
        
            TriIt tbegin() {
                return TriIt(nodes.begin(), nodes.end());
            }
        
            TriIt tend() {
                return TriIt(nodes.end(), nodes.end());
            }

            ConstTriIt ctbegin() const {
                return ConstTriIt(nodes.cbegin(), nodes.cend());
            }
        
            ConstTriIt ctend() const {
                return ConstTriIt(nodes.cend(), nodes.cend());
            }
            
            class ConstTriItNoSkip {
            private:
                typename std::map<Id, Diag>::const_iterator itnmin;
                typename std::map<Id, Diag>::const_iterator itnmax;
                const typename std::map<Id, Diag>::const_iterator itnend;
                typename std::map<Id, Real>::const_iterator ithhess;
                    
            public:            
                ConstTriItNoSkip(
                    const typename std::map<Id, Diag>::const_iterator &nodes_begin,
                    const typename std::map<Id, Diag>::const_iterator &nodes_end
                ): itnend {nodes_end} {
                    this->itnmin = nodes_begin;
                    this->itnmax = this->itnmin;
                    if (this->itnmax != this->itnend) {
                        ++(this->itnmax);
                        this->ithhess = this->itnmin->second.row.begin();
                    }
                }
                void operator++() {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::operator++: itnmax == end");
                    }
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
                decltype(ithhess->second) operator*() const {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::operator*: itnmax == end");
                    }
                    if (this->ithhess != this->itnmin->second.row.end() && this->ithhess->first == this->itnmax->first) {
                        return this->ithhess->second;
                    } else {
                        return 0;
                    }
                }
                Id id1() const {
                    if (this->itnmin == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::id1: itnmin == end");
                    }
                    return this->itnmin->first;
                }
                Id id2() const {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::id2 itnmax == end");
                    }
                    return this->itnmax->first;
                }
                const Diag &diag1() const {
                    if (this->itnmin == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::diag1: itnmin == end");
                    }
                    return this->itnmin->second;
                }
                const Diag &diag2() const {
                    if (this->itnmax == this->itnend) {
                        throw std::runtime_error("ConstTriItNoSkip::diag2: itnmax == end");
                    }
                    return this->itnmax->second;
                }
                inline friend bool operator!=(const ConstTriItNoSkip &it1, const ConstTriItNoSkip &it2) noexcept {
                    return it1.itnmin != it2.itnmin and it1.itnmax != it2.itnmax;
                }
            };
            
            ConstTriItNoSkip ctnbegin() const {
                return ConstTriItNoSkip(nodes.cbegin(), nodes.cend());
            }
        
            ConstTriItNoSkip ctnend() const {
                return ConstTriItNoSkip(nodes.cend(), nodes.cend());
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

