# uncertainties-cpp {#mainpage}

C++ header library for first-order uncertainty propagation. More or less a port
of the python package [uncertainties](https://github.com/lebigot/uncertainties)
(but see also [gvar](https://github.com/gplepage/gvar)).

## Installation

It is a header library so there is no need to compile and install a binary.
Place the directory `uncertainties` in a place of your choice (possibly
alongside your code) and make sure to include the special header
`uncertainties/impl.hpp` in one (and only one) of your source files.

The header `ureals.hpp` requires [Eigen](http://eigen.tuxfamily.org).

## Usage

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

## Documentation

Use `doxygen` to generate the documentation in html format. But it is just a
stub currently.

## Tests

Use `make` in the directory `test` to compile and run all the tests.

## License

This software is released under the GNU Lesser General Public license v3.0,
which means you can use it with differently licensed (eventually proprietary)
software provided that you release under the GPL/LGPL any modifications to this
library.
