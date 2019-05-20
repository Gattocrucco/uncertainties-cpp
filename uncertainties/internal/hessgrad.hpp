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
                Real dhess; // 2 x second derivative
                M7P<Real> mom;
            private:
                std::map<Id, Real> row;
                friend class HessGrad<Real>;
            };
        
        private:
            std::map<Id, Diag> nodes;
        
        public:
            inline bool at_most_one_grad() const noexcept {
                return this->nodes.size() == 0 || (this->nodes.size() == 1 && this->nodes.begin()->row.size() == 0);
            }
        
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
                    throw std::runtime_error("tri_get: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                Diag *n = this->nodes.find(minid);
                if (!n) {
                    return 0;
                }
                Real *t = n->row.find(maxid);
                if (!t) {
                    return 0;
                }
                Diag *N = this->nodes.find(maxid);
                if (!N) {
                    throw std::runtime_error("tri_get: !N");
                }
                return *t;
            }
        
            void tri(const Id id1, const Id id2, const Real &dhess) {
                if (id1 == id2) {
                    throw std::runtime_error("tri_set: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                Diag *n = this->nodes.find(minid);
                if (!n) {
                    throw std::runtime_error("tri_set: !n");
                }
                Diag *N = this->nodes.find(maxid);
                if (!N) {
                    throw std::runtime_error("tri_set: !N");
                }
                n->row[maxid] = dhess;
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
                RealIterator itdhess;
        
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
                        this->itdhess = this->itnmin->second.row.begin();
                        if (this->itdhess != this->itnmin->second.row.end()) {
                            this->update_itnmax(this->itdhess->first);
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
                        if (this->itdhess == this->itnmin->second.row.end()) {
                            throw std::runtime_error("TriIt::operator++: itdhess == end");
                        }
                        ++this->itdhess;
                        if (this->itdhess == this->itnmin->second.row.end()) {
                            ++this->itnmin;
                            this->update();
                        }
                    }
                }
                decltype(itdhess->second) operator*() const {
                    if (this->itdhess == this->itnmin->second.row.end()) {
                        throw std::runtime_error("TriIt::operator*: itdhess == end");
                    }
                    return this->itdhess->second;
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
                        if (it1.itdhess == it2.itdhess) {
                            throw std::runtime_error("TriIt::operator!=: itdhess1 == itdhess2");
                        }
                        return true;
                    }
                    if (it1.itdhess != it2.itdhess) {
                        throw std::runtime_error("TriIt::operator!=: itdhess1 != itdhess2");
                    }
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
            
            template<typename OtherReal>
            friend class HessGrad;
            
            template<typename OtherReal>
            operator HessGrad<OtherReal>() const {
                HessGrad<OtherReal> hg;
                for (const auto &it : this->nodes) {
                    typename HessGrad<OtherReal>::Diag &node = hg.nodes[it.first];
                    const Diag &diag = it.second;
                    M7P<OtherReal> mom(new std::array<OtherReal, 7>);
                    std::copy(diag.mom->begin(), diag.mom->end(), mom->begin());
                    node.grad = diag.grad;
                    node.dhess = diag.dhess;
                    node.mom = mom;
                    for (const auto &it2 : it.second.dhess) {
                        node.dhess[it2.first] = it2.second;
                    }
                }
                return hg;
            }
        };
    }
}

#endif /* end of include guard: UNCERTAINTIES_HESSGRAD_HPP_84C05D5B */

