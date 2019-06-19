from matplotlib import pyplot as plt
import uncertainties
import numpy as np

SIGMA_FACTOR = 5

table = np.load('fit.npy', mmap_mode='r')
N = len(table[0]['estimate'])

info = np.load('fit-info.npz')
true_par = info['true_par']

fig = plt.figure('fit')
fig.clf()
axs = fig.subplots(N, N)

def ms(s):
    return uncertainties.ufloat(np.mean(s), np.std(s) / np.sqrt(len(s)))

estimates = table['estimate'].T
biases = table['bias'].T
covs = table['cov'].T
sigmas = np.sqrt(np.einsum('iim->im', covs))

for i in range(N):
    ax = axs[i][i]
    
    noncorr = estimates[i]
    corr = estimates[i] - np.where(np.abs(biases[i]) < SIGMA_FACTOR * sigmas[i], biases[i], 0)
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
        
        ok = np.abs(biases[i]) < SIGMA_FACTOR * sigmas[i]
        ok &= np.abs(biases[j]) < SIGMA_FACTOR * sigmas[j]
        
        ax.plot((estimates[i] - biases[i])[ok], (estimates[j] - biases[j])[ok], 'x', label='corrected')
        ax.plot((estimates[i] - biases[i])[~ok], (estimates[j] - biases[j])[~ok], 'x', color='red')
        ax.plot(estimates[i][ok], estimates[j][ok], '.', markersize=2, label='not corrected')
        ax.plot(estimates[i][~ok], estimates[j][~ok], '.', markersize=2, color='red')
        ax.legend(loc='best', fontsize='small')

fig.show()
