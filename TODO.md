## UReal and UReal2

Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for what
`boost::mp` does and what `Eigen` expects.

Check if `UReal` works as user type to Eigen.

Python interface. (Standard C extension compiled with specific types?)

Use cholesky in `ureals`.

Find a C++ template library for complex numbers with arbitrary numerical type
(maybe in Boost?) (std::complex need to be defined only for builtin types) and
check UReal(2) works with that.

## UReal

Maybe allow higher moments with first order propagation? They are trivial, for example symmetric distributions will never produce asymmetries, but in general they may be useful
=> No because it changes too much UReal.

Add grad member function to compute derivatives.

Rewrite `UReal::cov` using synchronized iteration.

Since the gradient is already stored in a binary tree, maybe use balanced sum
to compute the standard deviation?

## UReal2

Check that `ureals` works with `UReal2`.

Make grad and hess publicly usable.

Write doc.

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

Fit with propagation like `lsqfit`. The gradient wrt data can be estimated
quickly using linear model approximation, is there something similar for the
second derivatives? Is the second order correction on least squares a good
bias correction in typical cases? Is it better to do ML with Edgeworth pdf (and
would it allow to preserve correlation with data)?
=> See notebook June 11

Move `std_moments` and `central_moments` to `UReal2` constructor with option to center moments.