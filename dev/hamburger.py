# See https://en.wikipedia.org/wiki/Hamburger_moment_problem

import sympy

N = 4

central_moments = (1, 0) + sympy.symbols(f'mu2:{2 * N + 1}')
std_moments = (0, 1) + sympy.symbols(f'k3:{2 * N + 1}')
sigma = sympy.symbols('sigma', positive=True)
central_as_std = tuple(std_moments[i] * sigma ** (i + 1) for i in range(len(std_moments)))

matrix = sympy.Matrix(N + 1, N + 1, lambda i, j: central_moments[i + j])

def collect(poly, variables, k=0):
    if k == len(variables):
        return sympy.poly(poly, variables).as_expr()
    quo, rem = sympy.div(sympy.poly(poly, variables[k]), sympy.poly(variables[k], variables[k]))
    if quo == 0:
        return collect(poly, variables, k + 1)
    elif sympy.poly(quo, variables).is_monomial:
        return collect(poly, variables, k + 1)
    else:
        return collect(quo, variables, k) * variables[k] + collect(rem, variables, k + 1)

dets = []
for m in range(2, N + 2):
    print(f'\nm = {m}')
    determinant = sympy.det(matrix[:m, :m])
    determinant = sympy.simplify(determinant)
    print('condition on central moments:')
    print(determinant)
    print('condition on central moments (collected):')
    print(collect(determinant, central_moments[2:]))
    determinant = determinant.subs(zip(central_moments[2:], central_as_std[1:]))
    determinant = sympy.simplify(determinant)
    determinant = determinant.subs(sigma, 1)
    print('condition on standardized moments:')
    print(determinant)
    dets.append(determinant)
    print('condition on standardized moments (collected):')
    print(collect(determinant, std_moments[2:]))
        
