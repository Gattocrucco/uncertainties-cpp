Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for what
`boost::mp` does and what `Eigen` expects.

`UReal::isindep` should return true if `id < 0` but `sigma.size() <= 1`.

Check if `UReal` works as user type to Eigen.

Third and fourth order correlation functions (automatical generation). First do it with non optimized iterations. Can I compute correlation functions using the mean correction of expressions?
=> No, example: E[xxx] = 0 if xxx is computed with UReal2

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Optimize `UReal2::binary` and `UReal::binary`.

Implement `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations
(implement `UReal(2)::unary_inplace`). (Second optional argument to all the
unary functions in `math.hpp`? => No because they have to return something in one case and nothing in the other.)

Edgeworth series to compute the pdf. Optionally allow to compress it so it stays
positive (exponential with crude estimation of the mode?). Generalize Edgeworth
series to higher dimensionality. Or, is maximum entropy viable if I have 3 and 4 point correlation functions?

Python interface. (Standard C extension compiled with specific types?)

Fit with propagation like `lsqfit`. The gradient wrt data can be estimated
quickly using linear model approximation, is there something similar for the
second derivatives? Is the second order correction on least squares a good
bias correction in typical cases? Is it better to do ML with Edgeworth pdf (and
would it allow to preserve correlation with data)?
=> See notebook June 11

Move `std_moments` and `central_moments` to `UReal2` constructor with option to center moments.