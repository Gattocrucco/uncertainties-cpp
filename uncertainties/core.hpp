#ifndef UNCERTAINTIES_CORE_HPP_D4C14D73
#define UNCERTAINTIES_CORE_HPP_D4C14D73

/*! \file
\brief Basic declarations and definitions; do not use this header directly.
*/

#include <string>

namespace uncertainties {
    namespace internal {
        using Id = int;
    
        extern Id next_id;
    }
    
    template<typename Real>
    class UReal;

    enum class Order {
        row_major,
        col_major
    };
    
    template<typename Number>
    inline const Number &nom(const Number &x) noexcept {
        return x;
    }
    
    template<typename Number>
    inline Number sdev(const Number &x) {
        return 0;
    }
    
    template<typename Number>
    std::string format(const Number &x,
                       const float errdig=1.5f,
                       const std::string &sep=" ± ");

    template<typename OutVector, typename InVectorA, typename InVectorB>
    OutVector ureals(const InVectorA &mu,
                     const InVectorB &cov,
                     const Order order=Order::row_major);
}

#endif /* end of include guard: UNCERTAINTIES_CORE_HPP_D4C14D73 */
