#include <memory>
#include <array>
#include <map>
#include <utility>

#include "core.hpp"

namespace uncertainties {
    template<typename Real, Prop prop>
    class UReal2 {
    private:
        struct Ivar {
            Real deriv;
            std::shared_ptr<std::array<Real, 7>> mom;
        };
        
        // variables
        std::map<std::pair<Id, Id>, Ivar> hess;
        std::map<Id, Ivar> grad;
        Real mu;
        std::array<Real, 3> mom;
        std::array<bool, 3> mom_cached;
        
    public:
        UReal2(const Real &mu, const std::array<Real, 7> &moments) {
            const Id id = ++internal::last_id;
            this->grad[id] = Ivar {Real(1), moments};
            this->mu = mu;
        }
        
        UReal2(const Real &mu) {
            this->mu = mu;
        }
        
        UReal2() {
            ;
        }
    };
    
    using udouble2e = UReal2<double, Prop::est>;
    using udouble2m = UReal2<double, Prop::mom>;
    
    using ufloat2e = UReal2<float, Prop::est>;
    using ufloat2m = UReal2<float, Prop::mom>;
}
