import sympy

k = sympy.symbols('K', integer=True)

def zero_moment(m):
    mom = 1;
    for i in range(m):
        mom *= k + 2 * i
    return mom

def central_mom_from_zero(zm):
    assert(len(zm) >= 2)
    assert(zm[0] == 1)
    cm = zm[:2]
    for n in range(2, len(zm)):
        c = 0
        for k in range(n + 1):
            c += sympy.binomial(n, k) * zm[k] * (-zm[1]) ** (n - k)
        cm.append(sympy.simplify(c))
    return cm

def central_moments(m):
    return central_mom_from_zero([zero_moment(i) for i in range(m + 1)])

def std_moments(m):
    cm = central_moments(m)
    if m >= 3:
        s = sympy.sqrt(cm[2])
    for i in range(3, len(cm)):
        cm[i] = sympy.simplify(cm[i] / s ** i)
    return cm
