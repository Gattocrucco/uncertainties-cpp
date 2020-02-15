# file fit-plot-hessian.py
# you can run this script while fit.py is running to see the partial results

from matplotlib import pyplot as plt
import numpy as np
import sys

l = len(sys.argv)
if l == 2:
    indices = [int(sys.argv[1])]
elif l == 3:
    indices = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
else:
    raise ValueError('Wrong arguments. Usage: <start> <end>, or <index>')

table = np.load('fit.npy', mmap_mode='r')
table = table[indices]
if not np.all(table['success']):
    raise RuntimeError(f'not all requested entries are complete')

info = np.load('fit-info.npz')
true_x = info['true_x']
true_y = info['true_y']

N = len(table[0]['estimate'])
fig = plt.figure('fit-plot-hessian')
fig.clf()
axs = fig.subplots(N + 1, len(indices), squeeze=False)

for (ipar, ientry), ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])

    if ipar == N:
        # ax.errorbar(
        #     info['true_x'], info['true_y'],
        #     xerr=1, yerr=1,
        #     linestyle='', marker='', color='red', alpha=0.3
        # )
        data_x = table[ientry]['data_x']
        data_y = table[ientry]['data_y']
        for x0, y0, x1, y1 in zip(data_x, data_y, true_x, true_y):
            ax.plot([x0, x1], [y0, y1], marker='', color='black', linewidth=1)
        ax.plot(data_x, data_y, linestyle='', marker='.', color='black')
        continue
    
    H = table[ientry]['stable_hessian'][ipar]
    amax = np.max(np.abs(H))
    ax.imshow(H, vmin=-amax, vmax=amax, cmap='PiYG')
    for fun in [ax.axvline, ax.axhline]:
        fun(9.5, linewidth=1, color='gray', linestyle=':')

fig.tight_layout()
fig.show()
