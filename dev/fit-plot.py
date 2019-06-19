from matplotlib import pyplot as plt
import uncertainties
import numpy as np

SIGMA_FACTOR = 4

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
altbiases = table['alt_bias'].T
covs = table['cov'].T
sigmas = np.sqrt(np.einsum('iim->im', covs))

ok = True
for i in range(N):
    ok &= np.abs(biases[i]) < SIGMA_FACTOR * sigmas[i]

altok = True
for i in range(N):
    altok &= np.abs(altbiases[i]) < SIGMA_FACTOR * sigmas[i]

for i in range(N):
    ax = axs[i][i]
    
    noncorr = estimates[i]
    corr = estimates[i] - np.where(ok, biases[i], 0)
    altcorr = estimates[i] - np.where(altok, altbiases[i], 0)
    ax.hist(
        noncorr, bins='auto', histtype='stepfilled',
        label='not corrected\n(bias = {})'.format(ms(noncorr - true_par[i])),
        color='lightgray', zorder=0, density=True
    )
    ax.hist(
        altcorr, bins='auto', histtype='step',
        label='reference correction\n(bias = {})'.format(ms(altcorr - true_par[i])),
        color='black', linestyle='--', zorder=1, density=True
    )
    ax.hist(
        corr, bins='auto', histtype='step',
        label='our correction\n(bias = {})'.format(ms(corr - true_par[i])),
        color='black', zorder=2, density=True
    )
    ax.plot(
        2 * [true_par[i]], ax.get_ylim(),
        scaley=False, label='true value', color='darkgray', linewidth=3,
        zorder=0.5
    )
    ax.legend(loc='best', fontsize='small')
    ax.set_title(f'Parameter {i}')
    ax.set_xlabel(f'parameter {i}')
    ax.set_ylabel('normalized counts')

for i in range(N):
    for j in range(i + 1, N):
        ax = axs[i][j]
        
        ax.plot(
            true_par[j], true_par[i],
            marker='+', linestyle='', color='black', markersize=8,
            zorder=5, label='true value'
        )
        ax.plot(
            estimates[j], estimates[i],
            marker='.', markersize=6, color='lightgray',
            linestyle='', label='not corrected'
        )
        ax.plot(
            (estimates[j] - biases[j])[ok], (estimates[i] - biases[i])[ok],
            marker='.', markersize=2, color='black',
            linestyle='', label='our correction'
        )
        ax.plot(
            (estimates[j] - biases[j])[~ok], (estimates[i] - biases[i])[~ok],
            marker='x', markersize=6, color='black',
            linestyle='', label='correction detected too large'
        )
        ax.legend(loc='best', fontsize='small')
        ax.set_title(f'Parameters $(0, 1)$')
        ax.set_xlabel(f'parameter {j}')
        ax.set_ylabel(f'parameter {i}')
    
for i in range(N):
    for j in range(i):
        ax = axs[i][j]
        
        k = i * N + j
        ax.errorbar(
            info['true_x'], info['true_y'],
            xerr=1, yerr=1,
            linestyle='', marker='', color='gray',
            label='true points'
        )
        ax.plot(
            table[k]['data_x'], table[k]['data_y'],
            linestyle='', marker='.', color='black', label='simulated data'
        )
        ax.legend(loc='best', fontsize='small')
        ax.set_title('Example dataset')
        ax.set_xlabel('t')
        ax.set_ylabel('x')

fig.tight_layout()
fig.show()
