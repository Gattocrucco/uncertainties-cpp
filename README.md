<!-- \mainpage -->
# uncertainties-cpp

<img src="doc/uncertainties-cpp-512.png" width="128px" height="128px"
style="display: none;" />
<img src="uncertainties-cpp-512.png" srcset="uncertainties-cpp-512.png 100w"
sizes="50px" style="float: right; left-margin: 1em;" />

C++ header library for first- and second-order uncertainty propagation. More or
less a port of the python packages
[uncertainties](https://github.com/lebigot/uncertainties) and
[soerp](https://github.com/tisimst/soerp) (but see also
[gvar](https://github.com/gplepage/gvar) and
[measurements.jl](https://github.com/JuliaPhysics/Measurements.jl)).

## Installation

It is a header library so there is no need to compile and install a binary.
Download the code from
[github](https://github.com/Gattocrucco/uncertainties-cpp),
place the directory `uncertainties` in a place of your choice (possibly
alongside your code) and make sure to include the special header
`uncertainties/impl.hpp` in one (and only one) of your source files.

### Requirements

The C++ dialect is C++11. The header `ureals.hpp` (not to be confused with
`ureal.hpp`) and the second-order propagation class require
[Eigen](http://eigen.tuxfamily.org).

## Usage

All the definitions are in the namespace `uncertainties`. The library is split
in various headers. For first-order propagation, the principal header is
`ureal.hpp` which defines the class template `UReal`, which is aliased to
`udouble = UReal<double>` and `ufloat = UReal<float>`. The header `ureal2.hpp`
defines the similar class `UReal2` that does second-order propagation.

Basic example:
~~~cpp
#include <iostream>
#include <uncertainties/ureal.hpp>
#include <uncertainties/io.hpp>
#include <uncertainties/impl.hpp>
namespace unc = uncertainties;
int main() {
    unc::udouble x(2, 1), y(2, 1);
    unc::udouble a = x - x;
    unc::udouble b = x - y;
    std::cout << a << ", " << b << "\n";
}
~~~

## Features

* Clear distinction between mean estimation and bias correction for second-order propagation.

* User-defined types supported.

* Same class for independent and dependent variables.

* First-order independent variables do not use the heap.

Note: complex numbers are not supported.

## Technical details

The gradients and hessians are computed with forward propagation. They are
always sparse, implemented with C++ maps (trees), and can be updated in-place.
Ab-initio covariance matrices are not supported, variables with arbitrary given
correlations can be created as linear transformations of independent variables.
This software is not appropriate for efficient numerical calculation, but won't
hang catastrophically.

## Documentation

The documentation is
[here](http://www.giacomopetrillo.com/software/uncertainties-cpp/doc/html). Use
`doxygen` in the `doc` directory to generate the documentation.

## Tests

Use `make` in the directory `test` to compile and run all the tests.

## License

This software is released under the GNU Lesser General Public license v3.0,
which means you can use it with differently licensed (eventually proprietary)
software provided that you release under the GPL/LGPL any modifications to this
library.
