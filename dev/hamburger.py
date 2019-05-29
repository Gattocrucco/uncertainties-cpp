# See https://en.wikipedia.org/wiki/Hamburger_moment_problem

import sympy

N = 4

central_moments = (1, 0) + sympy.symbols(f'mu2:{2 * N + 1}')
std_moments = (0, 1) + sympy.symbols(f'k3:{2 * N + 1}')
sigma = sympy.symbols('sigma', positive=True)
central_as_std = tuple(std_moments[i] * sigma ** (i + 1) for i in range(len(std_moments)))

matrix = sympy.Matrix(N + 1, N + 1, lambda i, j: central_moments[i + j])

for m in range(2, N + 2):
    determinant = sympy.det(matrix[:m, :m])
    determinant = sympy.simplify(determinant)
    determinant = determinant.subs(zip(central_moments[2:], central_as_std[1:]))
    determinant = sympy.simplify(determinant)
    print(determinant)
