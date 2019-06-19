import numpy as np
from matplotlib import pyplot as plt
import uncertainties

data = np.loadtxt('distre.txt')

true = data[0]
samples = data[1:]

moments = np.empty((samples.shape[1], 2));
moments[0, 0] = np.mean(samples[:, 0]);
moments[0, 1] = np.std(samples[:, 0]) / np.sqrt(len(samples));
for i in range(1, len(moments)):
    moments[i, 0] = np.sum((samples[:, 0] - moments[0, 0]) ** (i + 1)) / len(samples)
    moments[i, 1] = np.sqrt(np.sum(((samples[:, 0] - moments[0, 0]) ** (i + 1) - moments[i, 0]) ** 2)) / len(samples)

fig = plt.figure('distre')
fig.clf()

axs = np.concatenate(fig.subplots(2, 2));

for i in range(len(axs)):
    ax = axs[i]
    
    mean = np.mean(samples[:, i])
    std = np.std(samples[:, i]) / np.sqrt(len(samples))
    label = 'mean = {}'.format(uncertainties.ufloat(mean, std))
    mlabel = 'sample = {}'.format(uncertainties.ufloat(moments[i, 0], moments[i, 1]))
    
    ax.hist(samples[:, i], bins='auto', histtype='step')
    
    ax.plot(2 * [true[i]], ax.get_ylim(), scaley=False, label=f'true = {true[i]:.4g}')
    
    l, = ax.plot(2 * [moments[i, 0]], ax.get_ylim(), scaley=False, label=mlabel)
    l, = ax.plot(2 * [moments[i, 0] - moments[i, 1]], ax.get_ylim(), scaley=False, linestyle='--', color=l.get_color())
    ax.plot(2 * [moments[i, 0] + moments[i, 1]], ax.get_ylim(), scaley=False, linestyle=l.get_linestyle(), color=l.get_color())
    
    l, = ax.plot(2 * [mean], ax.get_ylim(), scaley=False, label=label)
    l, = ax.plot(2 * [mean - std], ax.get_ylim(), scaley=False, linestyle='--', color=l.get_color())
    ax.plot(2 * [mean + std], ax.get_ylim(), scaley=False, linestyle=l.get_linestyle(), color=l.get_color())
    
    ax.legend(loc='best', fontsize='small')

fig.show()
