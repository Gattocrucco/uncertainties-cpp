#include <memory>
#include <array>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <cmath>

#include "core.hpp"

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
            };
        
        private:
            struct Node {
                Diag d;
                std::map<Id, Real> dhess;
            };
        
            std::map<Id, Node> nodes;
        
        public:
            inline bool at_most_one_grad() const noexcept {
                return this->nodes.size() == 0 || (this->nodes.size() == 1 && this->nodes.begin()->dhess.size() == 0);
            }
        
            inline typename std::map<Id, Node>::size_type size() const noexcept {
                return this->nodes.size();
            }
        
            inline const Diag &diag(const Id id) const noexcept {
                return this->nodes.at(id).d;
            }
        
            inline Diag &diag(const Id id) noexcept {
                return this->nodes[id].d;
            }
        
            const Real &tri(const Id id1, const Id id2) const {
                if (id1 == id2) {
                    throw std::runtime_error("tri_get: id1 == id2");
                }
                const Id minid = std::min(id1, id2);
                const Id maxid = std::max(id1, id2);
                Node *n = this->nodes.find(minid);
                if (!n) {
                    return 0;
                }
                Real *t = n->dhess.find(maxid);
                if (!t) {
                    return 0;
                }
                Node *N = this->nodes.find(maxid);
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
                Node *n = this->nodes.find(minid);
                if (!n) {
                    throw std::runtime_error("tri_set: !n");
                }
                Node *N = this->nodes.find(maxid);
                if (!N) {
                    throw std::runtime_error("tri_set: !N");
                }
                n->dhess[maxid] = dhess;
            }
        
            class DiagIt {
            private:
                typename std::map<Id, Node>::iterator it;
            
            public:                
                DiagIt(const typename std::map<Id, Node>::iterator &i): it {i} {
                    ;
                }
                inline void operator++() noexcept {
                    ++it;
                }
                inline std::pair<Id, Diag> &operator*() const noexcept {
                    return static_cast<std::pair<Id, Diag> &>(*it);
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
                typename std::map<Id, Node>::const_iterator it;
            
            public:
                ConstDiagIt(const typename std::map<Id, Node>::const_iterator &i): it {i} {
                    ;
                }
                inline void operator++() noexcept {
                    ++it;
                }
                inline const std::pair<Id, Diag> &operator*() const noexcept {
                    return static_cast<const std::pair<Id, Diag> &>(*it);
                }
                friend inline bool operator!=(const ConstDiagIt &it1, const ConstDiagIt &it2) {
                    return it1.it != it2.it;
                }
            };
        
            ConstDiagIt cdbegin() const {
                return ConstDiagIt(nodes.begin());
            }
        
            ConstDiagIt cdend() const {
                return ConstDiagIt(nodes.end());
            }

            template<typename NodeIterator, typename RealIterator>
            class TriItTempl {
            private:
                NodeIterator itnmin;
                NodeIterator itnmax;
                const NodeIterator itnend;
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
                        this->itdhess = this->itnmin->second.dhess.begin();
                        if (this->itdhess != this->itnmin->second.dhess.end()) {
                            this->update_itnmax(this->itdhess->first);
                        }
                    }
                }
            
                void throw_on_end() const {
                    if (this->itnmin == this->itnend) {
                        throw std::runtime_error("TriIt::operator*: itnmin == end");
                    } else if (this->itdhess == this->itnmin->second.dhess.end()) {
                        throw std::runtime_error("TriIt::operator*: itdhess == end");
                    }
                }
            
            public:
                using RefType = decltype(RealIterator()->second);
                static_assert(std::is_lvalue_reference<RefType>::value, "RefType not a reference");
            
                TriItTempl(const NodeIterator &nodes_begin,
                           const NodeIterator &nodes_end):
                itnend {nodes_end} {
                    this->itnmin = nodes_begin;
                    this->update();
                }
                void operator++() {
                    if (this->itnmin != this->itnend) {
                        ++this->itdhess;
                        if (this->itdhess == this->itnmin->second.dhess.end()) {
                            ++this->itnmin;
                            this->update();
                        }
                    }
                }
                RefType operator*() const {
                    this->throw_on_end();
                    return this->itdhess->second;
                }
                Id id1() const {
                    this->throw_on_end();
                    return this->itnmin->first;
                }
                Id id2() const {
                    this->throw_on_end();
                    return this->itnmax->first;
                }
                const Diag &diag1() const {
                    this->throw_on_end();
                    return this->itnmin->second.d;
                }
                const Diag &diag2() const {
                    this->throw_on_end();
                    return this->itnmax->second.d;
                }
                using ThisType = TriItTempl<NodeIterator, RealIterator>;
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
        
            using TriIt = TriItTempl<typename std::map<Id, Node>::iterator, typename std::map<Id, Node>::iterator>;
            using ConstTriIt = TriItTempl<typename std::map<Id, Node>::const_iterator, typename std::map<Id, Node>::const_iterator>;
        
            TriIt tbegin() {
                return TriIt(nodes.begin(), nodes.end());
            }
        
            TriIt tend() {
                return TriIt(nodes.end(), nodes.end());
            }

            ConstTriIt ctbegin() const {
                return ConstTriIt(nodes.begin(), nodes.end());
            }
        
            ConstTriIt ctend() const {
                return ConstTriIt(nodes.end(), nodes.end());
            }
            
            template<typename OtherReal>
            friend class HessGrad;
            
            template<typename OtherReal>
            operator HessGrad<OtherReal>() const {
                HessGrad<OtherReal> hg;
                for (const auto &it : this->nodes) {
                    typename HessGrad<OtherReal>::Node &node = hg.nodes[it.first];
                    Diag diag = it.second.d;
                    M7P<OtherReal> mom(new std::array<OtherReal, 7>);
                    std::copy(diag.mom->begin(), diag.mom->end(), mom->begin());
                    node.d = HessGrad<OtherReal>::Diag(diag.grad, diag.dhess, mom);
                    for (const auto &it2 : it.second.dhess) {
                        node.dhess[it2.first] = it2.second;
                    }
                }
                return hg;
            }
        };
        
        template<typename Real>
        Real compute_mom(const HessGrad<Real> &hg, const int n) {
            if (n < 2 or n > 4) {
                throw std::invalid_argument("uncertainties::internal::compute_mom: n not in [2, 4]");
            }
            using ConstDiagIt = typename HessGrad<Real>::ConstDiagIt;
            using ConstTriIt = typename HessGrad<Real>::ConstTriIt;
            using Diag = typename HessGrad<Real>::Diag;
            Real m(0);
            if (n == 2) {
                // formula:
                // C[y^2] =
                // G_i^2 V_{ii} +
                // 2 G_i H_{ii} V_{iii} +
                // H_{ii} H_{ii} V_{iiii} +
                // 2 \sum_{i < j} (H_{ii} H_{jj} + 2 H_{ij}^2) V_{ii} V_{jj}
                for (ConstDiagIt it = hg.cdbegin(); it != hg.cdend(); ++it) {
                    const Diag &d = it.second;
                    m += d.grad * d.grad * v<2>(d.mom);
                    m += 2 * d.grad * d.dhess * v<3>(d.mom);
                    m += d.dhess * d.dhess * v<4>(d.mom);
                }
                for (ConstTriIt it = hg.ctbegin(); it != hg.ctend(); ++it) {
                    const Diag &d1 = it.diag1();
                    const Diag &d2 = it.diag2();
                    const Real &dhess = *it;
                    m += 2 * (d1.dhess * d2.dhess + 2 * dhess * dhess) * v<2>(d1.mom) * v<2>(d2.mom);
                }
            }
            return m;
        }
    }
    
    template<typename Real, Prop prop>
    class UReal2 {
    private:
        using HessGrad = internal::HessGrad<Real>;
        using Diag = typename HessGrad::Diag;
        using ConstDiagIt = typename HessGrad::ConstDiagIt;
        using ConstTriIt = typename HessGrad::ConstTriIt;
        
        // variables
        HessGrad hg;
        Real mu;
        std::array<Real, 3> mom;
        std::array<bool, 3> mom_cached;
        
    public:
        UReal2(const Real &mu, const std::array<Real, 7> &moments) {
            const Id id = ++internal::last_id;
            this->mu = mu;
            this->hg.diag(id) = Diag(1, 0, new std::array<Real, 7>(moments));
        }

        UReal2(const Real &mu) {
            this->mu = mu;
        }

        UReal2() {
            ;
        }
        
        inline bool isindep() const noexcept {
            return this->hg.at_most_one_grad();
        }

        Id indepid() const noexcept {
            if (not this->isindep()) {
                return invalid_id;
            } else if (this->hg.size() == 1) {
                return this->hg.dbegin()->first;
            } else {
                return 0;
            }
        }

        inline const Real &n() const noexcept {
            return this->mu;
        }

        const Real &s() {
            using std::sqrt;
            return sqrt(this->m(2));
        }

        const Real &s() const {
            using std::sqrt;
            return sqrt(this->m(2));
        }
        
        const Real &skew() {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        const Real &skew() const {
            return this->m(3) / (this->s() * this->m(2));
        }
        
        const Real &kurt() {
            const Real &s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        const Real &kurt() const {
            const Real &s2 = this->m(2);
            return this->m(4) / (s2 * s2);
        }
        
        const Real &m(const int n) {
            const int i = n - 2;
            if (not this->mom_cached.at(i)) {
                this->mom[i] = internal::compute_mom(this->hg, n);
                this->mom_cached[i] = true;
            }
            return this->mom[i];
        }
        
        const Real &m(const int n) const {
            const int i = n - 2;
            if (not this->mom_cached.at(i)) {
                return internal::compute_mom(this->hg, n);
            } else {
                return this->mom[i];
            }
        }

        template<typename... Args>
        std::string format(Args &&... args) const {
            return uncertainties::format(*this, std::forward<Args>(args)...);
        }
        
        friend UReal2<Real, prop> change_nom(const UReal2<Real, prop> &x, const Real &n) {
            UReal2<Real, prop> y = x;
            y.mu = n;
            return y;
        }
        
        template<typename OtherReal, Prop other_prop>
        friend class UReal2;
        
        template<typename OtherReal>
        operator UReal2<OtherReal, prop>() const {
            UReal2<OtherReal, prop> x;
            x.hg = this->hg;
            x.mu = this->mu;
            std::copy(this->mom.begin(), this->mom.end(), x.mom.begin());
            x.mom_cached = this->mom_cached;
            return x;
        }
        
        template<Prop other_prop>
        explicit UReal2(const UReal2<Real, other_prop> &x) {
            this->hg = x.hg;
            this->mu = x.mu;
            std::copy(x.mom.begin(), x.mom.end(), this->mom.begin());
            this->mom_cached = x.mom_cached;
        }
        
        friend Real cov(const UReal2<Real, prop> &x, const UReal2<Real, prop> &y) {
            return 0;
        }
    };
    
    using udouble2e = UReal2<double, Prop::est>;
    using udouble2m = UReal2<double, Prop::mom>;
    
    using ufloat2e = UReal2<float, Prop::est>;
    using ufloat2m = UReal2<float, Prop::mom>;
}
