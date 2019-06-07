import sympy

def pdf(x):
    return sympy.Piecewise(
        (0, x < 0),
        (2 * x, (0 < x) & (x < 1)),
        (0, x > 1)
    )

def pdfm(m):
    return sympy.Piecewise(
        (0, m < 0),
        (-4 * m * sympy.log(m), (0 < m) & (m < 1)),
        (0, m > 1)
    )

def mom(n, pdf, s=1, *args):
    x = sympy.symbols('x')
    norm = sympy.integrate(pdf(x, *args), (x, -sympy.oo, sympy.oo))
    mu = sympy.integrate(pdf(x, *args) * x, (x, -sympy.oo, sympy.oo))
    m = [norm, s * mu]
    for i in range(2, n + 1):
        m.append(sympy.simplify(sympy.integrate(pdf(x + mu, *args) * x ** i, (x, -sympy.oo, sympy.oo)) * s ** i))
    return m

