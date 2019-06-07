import sympy

def pdf(x, w, l1, l2):
    # w in [0, 1]
    # l1, l2 >= 0
    return sympy.Piecewise(
        (0, x < -l1),
        (w / l1, (-l1 <= x) & (x < 0)),
        ((1 - w) / l2, (0 <= x) & (x < l2)),
        (0, l2 <= x)
    )

def zero_moment(n, w, l1, l2):
    x = sympy.symbols('x', real=True)
    return sympy.simplify(sympy.integrate(pdf(x, w, l1, l2) * x ** n, (x, -l1, l2)))

w, l1, l2 = sympy.symbols('w, delta1, delta2', positive=True)
x = sympy.symbols('x', real=True)
