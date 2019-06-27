## Generic

See how to put package on homebrew.

## UReal and UReal2

Function to parse strings.

In math functions with singularities in the derivative, check that the mean is
not close to the singularity and warn the user on cerr. => No because it
would require to compute the standard deviation at each step.

Implement `numeric_limits`. In namespace `uncertainties` or `std`? Look for what
`boost::mp` does and what `Eigen` expects.

Check if `UReal` works as user type to Eigen.

Python interface. (Standard C extension compiled with specific types?)

Find a C++ template library for complex numbers with arbitrary numerical type
(maybe in Boost?) (std::complex need to be defined only for builtin types) and
check UReal(2) works with that.

Use the `UNCERTAINTIES_EXTERN_*` macros in all headers, and for friend
functions.

## UReal

Use cholesky in `ureals`.

Use balanced sum to compute the standard deviation?

## UReal2

Allow first order propagation for all moments with function `first_order_m()`.

Implement bias correction for higher order moments.

Design a version of `ureals` for `UReal2`. Linear transformation? Same
standardized moments for all variables?

Third and fourth order correlation functions (automatical generation). Do it
first storing all ids and coefficients in arrays then iterating.

Check higher order correlation functions using relations with lower order
correlations if there are identical arguments.

Implement `UReal2::binary_assign`.

Think a sensible interface to allow inplace unary operations (implement
`UReal(2)::unary_inplace`). (Second optional argument to all the unary
functions in `math.hpp`? => No because they have to return something in one
case and nothing in the other.)

Maximum entropy pdf given the moments. See RV Abramov paper.

Fit with propagation like `lsqfit` but at second order.

Move `std_moments` and `central_moments` to `UReal2` constructor with option to
center moments.
