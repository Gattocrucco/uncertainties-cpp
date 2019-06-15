## Generic

See how to put package on homebrew.

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

Remove specializations with `float` since probably it suffers from precision
problems. (Or not?)

Use the `UNCERTAINTIES_EXTERN_*` macros in all headers.

## UReal

Maybe allow higher moments with first order propagation? They are trivial, for example symmetric distributions will never produce asymmetries, but in general they may be useful
=> No because it changes too much UReal.

Add grad member function to compute derivatives.

Rewrite `UReal::cov` using synchronized iteration.

Since the gradient is already stored in a binary tree, maybe use balanced sum
to compute the standard deviation?

## UReal2

`corr`

Remove moment caching altogether?

Rescale internally standardized moments such that the 8th std moment of a
normal distribution is 1, this should increase numerical stability. The
rescaling should have few digits if written in base 2. The scaling can be
inverted just before returning from `UReal::m()`. => No because the sigma is in
the gradient and hessian and having different sigmas around will make the change
ineffective.

Check that `ureals` works with `UReal2`.
=> It will not by design, higher order moments must be specified.

Make grad and hess publicly usable.

Third and fourth order correlation functions (automatical generation). Do it
first storing all coefficients in arrays then iterating.

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Optimize `UReal2::binary` and `UReal::binary`.

Implement `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations
(implement `UReal(2)::unary_inplace`). (Second optional argument to all the
unary functions in `math.hpp`? => No because they have to return something in one case and nothing in the other.)

Maximum entropy pdf given the moments. See RV Abramov paper.

Fit with propagation like `lsqfit` but at second order. See notebook June 11.

Move `std_moments` and `central_moments` to `UReal2` constructor with option to center moments.
