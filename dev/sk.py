import sys
from scipy import optimize, integrate, special
import numpy as np
from matplotlib import pyplot as plt
import numba

skew = float(sys.argv[1])
kurt = float(sys.argv[2])

assert(-skew ** 2 + kurt - 1 >= 0)

## least squares to find max entropy solution

def p(x, l0, l1, l2, l3, l4):
    return np.exp(-(l0 + x * (l1 + x * (l2 + x * (l3 + x * l4)))))

@numba.jit(nopython=True)
def integrand(x, mu3, mu4, l1, l2, l3, l4):
    return np.exp(-(
        l1 * x + 
        l2 * (x**2 - 1) +
        l3 * (x**3 - mu3) +
        l4 * (x**4 - mu4)
    ))

@numba.jit(nopython=True)
def grad_term(x, n, mu3, mu4):
    if n == 1:
        return 0 - x
    elif n == 2:
        return 1 - x**2
    elif n == 3:
        return mu3 - x**3
    elif n == 4:
        return mu4 - x**4

@numba.jit(nopython=True)
def grad_integrand(x, n, mu3, mu4, l1, l2, l3, l4):
    return integrand(x, mu3, mu4, l1, l2, l3, l4) * grad_term(x, n, mu3, mu4)

@numba.jit(nopython=True)
def hess_integrand(x, n, m, mu3, mu4, l1, l2, l3, l4):
    return integrand(x, mu3, mu4, l1, l2, l3, l4) * grad_term(x, n, mu3, mu4) * grad_term(x, m, mu3, mu4)

def fun(l):
    return integrate.quad(integrand, -np.inf, np.inf, args=(skew, kurt, *l))[0]

def jac(l):
    return np.array([integrate.quad(grad_integrand, -np.inf, np.inf, args=(n, skew, kurt, *l))[0] for n in (1, 2, 3, 4)])

def hess(l):
    h = np.empty((4, 4))
    for i in range(4):
        for j in range(i, 4):
            h[i, j] = integrate.quad(hess_integrand, -np.inf, np.inf, args=(i + 1, j + 1, skew, kurt, *l))[0]
            h[j, i] = h[i, j]
    return h

l0 = [1, 1, 1, 1]

result = optimize.minimize(
    fun, l0,
    jac=jac,
    hess=hess,
    method='trust-constr',
    options=dict(disp=True),
    bounds=[(None, None), (None, None), (None, None), (0, None)]
)
print(result)

N = integrate.quad(p, -np.inf, np.inf, args=(1, *result.x))[0]

## edgeworth series

k3 = skew
k4 = kurt - 3

def gca(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1/2 * x**2) * (
        1 +
        k3 / (2 * 3) * special.eval_hermitenorm(3, x) +
        k4 / (2 * 3 * 4) * special.eval_hermitenorm(4, x)
    )

## plot

fig = plt.figure('sk')
fig.clf()
ax = fig.subplots(1, 1)

x = np.linspace(-5, 5, 1000)
ax.plot(x, p(x, 1, *result.x) / N, label='max entropy')
ax.plot(x, gca(x), label='Gram-Charlier A')
ax.legend(loc='best')

fig.show()
