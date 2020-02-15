# file fit.py

from scipy import optimize, linalg
import autograd
from autograd import numpy as np
from numpy.lib import format as nplf
import progressbar

M = 1000 # number of monte carlo
N = 2 # number of parameters
def mu(x, p):
    return p[0] * np.cos(x / p[1])
true_x = np.linspace(0, 30, 10)
true_par = np.array([10, 4])

##################################

table = nplf.open_memmap('fit.npy', mode='w+', shape=(M,), dtype=[
    ('success', bool),
    ('estimate', float, N),
    ('bias', float, N),
    ('standard_bias', float, N),
    ('cov', float, (N, N)),
    ('standard_cov', float, (N, N)),
    ('data_y', float, len(true_x)),
    ('data_x', float, len(true_x)),
    ('complete_estimate', float, N + len(true_x)),
    ('complete_bias', float, N + len(true_x)),
    ('complete_cov', float, (N + len(true_x), N + len(true_x))),
    ('hessian', float, (N + len(true_x), 2 * len(true_x), 2 * len(true_x))),
    ('stable_hessian', float, (N + len(true_x), 2 * len(true_x), 2 * len(true_x)))
])
table['success'] = False

def res(p, data):
    par = p[:N]
    x = p[N:]
    data_x = data[:len(data) // 2]
    data_y = data[len(data) // 2:]
    return np.concatenate([data_y - mu(x, par), data_x - x])

jac = autograd.jacobian(res, 0)

def Q(p, data):
    r = res(p, data)
    return np.sum(r ** 2)

f = autograd.jacobian(Q, 0)
dfdy = autograd.jacobian(f, 1)
dfdp = autograd.jacobian(f, 0)
dfdpdy = autograd.jacobian(dfdp, 1)
dfdpdp = autograd.jacobian(dfdp, 0)

true_y = mu(true_x, true_par)

np.savez(
    'fit-info.npz',
    true_x=true_x,
    true_par=true_par,
    true_y=true_y
)

def least_squares_cov(result):
    _, s, VT = linalg.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    return np.dot(VT.T / s**2, VT)

def compute_bias_cov(result, data):
    dfdy_ = dfdy(result.x, data)
    dfdp_ = dfdp(result.x, data)
    dfdpdy_ = dfdpdy(result.x, data)
    dfdpdp_ = dfdpdp(result.x, data)
    
    grad = np.linalg.solve(dfdp_, -dfdy_)
    assert(grad.shape == (N + len(true_x), 2 * len(true_x)))
    
    cov = np.einsum('ai,bi->ab', grad, grad)
    assert(np.allclose(cov, cov.T))
    
    B = (
        - np.einsum('abi,bj->aij', dfdpdy_, grad)
        - np.einsum('abi,bj->aji', dfdpdy_, grad)
        - np.einsum('abg,bi,gj->aij', dfdpdp_, grad, grad)
    )
    assert(B.shape == (N + len(true_x), 2 * len(true_x), 2 * len(true_x)))
    B_ = B.reshape(N + len(true_x), 4 * len(true_x) * len(true_x))

    hess = np.linalg.solve(dfdp_, B_)
    hess = hess.reshape(N + len(true_x), 2 * len(true_x), 2 * len(true_x))
    assert(np.allclose(hess, np.einsum('aji', hess)))

    return 1/2 * np.einsum('aii', hess), cov, hess

for i in progressbar.progressbar(range(M)):
    data_x = true_x + np.random.randn(len(true_x))
    data_y = true_y + np.random.randn(len(true_x))
    data = np.concatenate([data_x, data_y])
    
    p0 = np.concatenate([true_par, true_x])
    result = optimize.least_squares(res, p0, jac=jac, args=(data,))
    if not result.success:
        print(f'minimization failed for i = {i}')
        continue
    
    bias, cov, hessian = compute_bias_cov(result, data)
    zero_residual_data = np.concatenate([
        result.x[N:],
        mu(result.x[N:], result.x[:N])
    ])
    standard_bias, _, stable_hessian = compute_bias_cov(result, zero_residual_data)
        
    table[i]['estimate'] = result.x[:N]
    table[i]['bias'] = bias[:N]
    table[i]['standard_bias'] = standard_bias[:N]
    table[i]['cov'] = cov[:N, :N]
    table[i]['data_y'] = data_y
    table[i]['data_x'] = data_x
    table[i]['complete_estimate'] = result.x
    table[i]['complete_bias'] = bias
    table[i]['complete_cov'] = cov
    table[i]['standard_cov'] = least_squares_cov(result)[:N, :N]
    table[i]['hessian'] = hessian
    table[i]['stable_hessian'] = stable_hessian
    table[i]['success'] = result.success

    table.flush()

del table
