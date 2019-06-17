from matplotlib import pyplot as plt
import numpy as np

even_moments = np.cumprod(np.concatenate([[1], np.arange(1, 18, 2)]))
var = even_moments[:-1:2] - even_moments[:len(even_moments) // 2] ** 2
N = var / even_moments[:len(var)] ** 2

fig = plt.figure('momest')
fig.clf()
ax = fig.subplots(1, 1)

ax.plot(np.arange(len(var)) * 2, N, '.')
ax.grid()
ax.set_xlabel('moment order')
ax.set_ylabel('sample required for 100 % error on moment')

fig.show()
