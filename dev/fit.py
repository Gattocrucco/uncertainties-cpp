from scipy import optimize
import autograd
from autograd import numpy as np
from matplotlib import pyplot as plt
import progressbar
import uncertainties

M = 2 # number of monte carlo
N = 2 # number of parameters
def mu(x, p):
    return p[0] * np.cos(x / p[1])
true_x = np.linspace(0, 30, 10)
true_par = np.array([10, 4])

##########################################################

estimates = np.empty((N, M))
biases = np.empty((N, M))
covs = np.empty((N, N, M))

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

for i in progressbar.progressbar(range(M)):
    data_x = true_x + np.random.randn(len(true_x))
    data_y = mu(true_x, true_par) + np.random.randn(len(true_x))
    data = np.concatenate([data_x, data_y])

    p0 = np.concatenate([true_par, true_x])
    result = optimize.least_squares(res, p0, jac=jac, args=(data,))
    assert(result.success)
    
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

    hess = np.linalg.solve(dfdp_, B_).reshape(N + len(true_x), 2 * len(true_x), 2 * len(true_x))
    assert(np.allclose(hess, np.einsum('aji', hess)))

    bias = 1/2 * np.einsum('aii', hess)
    
    estimates[:, i] = result.x[:N]
    biases[:, i] = bias[:N]
    covs[..., i] = cov[:N, :N]

sigmas = np.sqrt(np.einsum('iim->im', covs))

fig = plt.figure('fit')
fig.clf()
axs = fig.subplots(N, N)

def ms(s):
    return uncertainties.ufloat(np.mean(s), np.std(s) / np.sqrt(len(s)))

for i in range(N):
    ax = axs[i][i]
    
    noncorr = estimates[i]
    corr = estimates[i] - np.where(np.abs(biases[i]) < sigmas[i], biases[i], 0)
    ax.hist(
        noncorr, bins='auto', histtype='step',
        label='not corrected\nbias = {}'.format(ms(noncorr - true_par[i]))
    )
    ax.hist(
        corr, bins='auto', histtype='step',
        label='corrected\nbias = {}'.format(ms(corr - true_par[i]))
    )
    ax.plot(2 * [true_par[i]], ax.get_ylim(), scaley=False, label='true')
    ax.legend(loc='best', fontsize='small')

for i in range(N):
    for j in range(i + 1, N):
        ax = axs[i][j]
        
        ok = np.abs(biases[i]) < sigmas[i]
        ok &= np.abs(biases[j]) < sigmas[j]
        
        ax.plot((estimates[i] - biases[i])[ok], (estimates[j] - biases[j])[ok], 'x', label='corrected')
        ax.plot((estimates[i] - biases[i])[~ok], (estimates[j] - biases[j])[~ok], 'x', color='red')
        ax.plot(estimates[i][ok], estimates[j][ok], '.', markersize=2, label='not corrected')
        ax.plot(estimates[i][~ok], estimates[j][~ok], '.', markersize=2, color='red')
        ax.legend(loc='best', fontsize='small')

fig.show()
