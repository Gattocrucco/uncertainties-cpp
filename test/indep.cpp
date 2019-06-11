#include <cassert>

#include <uncertainties/ureal.hpp>

namespace unc = uncertainties;

int main() {
    unc::udouble x;
    assert(x.isindep());
    x = 1;
    assert(x.isindep());
    x = unc::udouble {1, 0.1};
    assert(x.isindep());
    unc::udouble y {0.4, 0.2};
    unc::udouble z = x + y;
    assert(not z.isindep());
    assert(z.indepid() == unc::invalid_id);
    y = x + x;
    assert(y.isindep());
    assert(x.indepid() == y.indepid());
    assert(corr(x, y) == 1);
}
