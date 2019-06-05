Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for what
`boost::mp` does and what `Eigen` expects.

`UReal::isindep` should return true if `id < 0` but `sigma.size() <= 1`.

Check if `UReal` works as user type to Eigen.

fourth order momentum

third and fourth order correlation functions

check higher order correlation functions using relations with lower order
correlations if there are identical arguments

optimize `UReal2::binary` and `UReal::binary`

implement `UReal2::binary_assign`

think a sensible interface to allow inplace unary operations
(implement `UReal(2)::unary_inplace`). (Second optional argument to all the
unary functions in `math.hpp`?)

Edgeworth series to compute the pdf. Optionally allow to compress it so it stays
positive (exponential with crude estimation of the mode?). Generalize Edgeworth
series to higher dimensionality.

Python interface. (Standard C extension compiled with specific types?)
