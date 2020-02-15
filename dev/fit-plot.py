# file fit-plot.py
# you can run this script while fit.py is running to see the partial results

from matplotlib import pyplot as plt
import uncertainties
import numpy as np

SIGMA_FACTOR = 5

table = np.load('fit.npy', mmap_mode='r')
table = table[table['success']]
N = len(table[0]['estimate'])

info = np.load('fit-info.npz')
true_par = info['true_par']

fig = plt.figure('fit')
fig.clf()
axs = fig.subplots(N + 1, N)

def ms(s):
    return uncertainties.ufloat(np.mean(s), np.std(s) / np.sqrt(len(s)))

estimates = table['estimate'].T
biases = table['bias'].T
sbias = table['standard_bias'].T
covs = table['cov'].T
scovs = table['standard_cov'].T
sigmas = np.sqrt(np.einsum('iim->im', covs))
ssigmas = np.sqrt(np.einsum('iim->im', scovs))

ok = np.einsum(
    'ua,uab,ub->u',
    table['bias'], np.linalg.inv(table['standard_cov']), table['bias']
) < SIGMA_FACTOR ** 2

for i in range(N):
    ax = axs[i][i]
    
    noncorr = estimates[i]
    corr = estimates[i] - np.where(ok, biases[i], 0)
    stdcorr = estimates[i] - sbias[i]
    ax.hist(
        noncorr, bins='auto', histtype='stepfilled',
        label='least squares...\n(bias = {},\nstd = {:.2g})'.format(
            ms(noncorr - true_par[i]),
            np.std(noncorr)
        ),
        color='lightgray', zorder=0, density=True
    )
    ax.hist(
        corr, bins='auto', histtype='step',
        label='...with correction (12)\n(bias = {},\nstd = {:.2g})'.format(
            ms(corr - true_par[i]),
            np.std(corr)
        ),
        color='black', zorder=2, density=True
    )
    ax.hist(
        stdcorr, bins='auto', histtype='step',
        label='...with correction (34)\n(bias = {},\nstd = {:.2g})'.format(
            ms(stdcorr - true_par[i]),
            np.std(stdcorr)
        ),
        linestyle='--', color='black', zorder=2, density=True
    )
    ax.plot(
        2 * [true_par[i]], ax.get_ylim(),
        scaley=False, label='true value', color='darkgray', linewidth=3,
        zorder=0.5
    )
    ax.legend(loc='best', fontsize='small')
    ax.set_title(f'Parameter {i} estimator')
    ax.set_xlabel(f'parameter {i}')
    ax.set_ylabel('normalized counts')
    
    # now we plot histogram of estimated standard deviation
    ax = axs[N][i]
    
    sstd = np.std(corr)
    sstdnc = np.std(noncorr)
    
    ax.hist(
        ssigmas[i],
        bins='auto', histtype='stepfilled', density=True,
        color='lightgray', zorder=0,
        label='eq. 27'
    )
    ax.hist(
        np.where(ok, sigmas[i], ssigmas[i]),
        bins='auto', histtype='step', density=True,
        color='black', linestyle='-', zorder=1,
        label='eq. 33\n(or eq. 27 when correction too large)'
    )
    ax.plot(
        2 * [sstdnc], ax.get_ylim(),
        scaley=False, color='darkgray', linewidth=3, zorder=0.5,
        label=f'sample sdev of noncorrected ({sstdnc:.2g})'
    )
    ax.plot(
        2 * [sstd], ax.get_ylim(),
        scaley=False, color='black', zorder=1.5, linestyle='-',
        label=f'sample sdev w. correction (12) ({sstd:.2g})'
    )
    ax.legend(loc='best', fontsize='small')
    ax.set_title(f'Standard deviation of parameter {i} est.')
    ax.set_xlabel(f'sdev of parameter {i}')
    ax.set_ylabel('normalized counts')

for i in range(N):
    for j in range(i + 1, N):
        ax = axs[i][j]
        
        ax.plot(
            true_par[j], true_par[i],
            marker='+', linestyle='', color='red', markersize=8,
            zorder=5, label='true value'
        )
        ax.plot(
            estimates[j], estimates[i],
            marker='.', markersize=6, color='lightgray',
            linestyle='', label='least squares'
        )
        ax.plot(
            (estimates[j] - biases[j])[ok], (estimates[i] - biases[i])[ok],
            marker='.', markersize=2, color='black',
            linestyle='', label='correction (12)'
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
        
        k = j * N + i
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
