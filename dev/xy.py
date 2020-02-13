import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

x = np.random.randn(3, 1000000)

# z = np.sign(y) * np.log(1 + np.abs(y))
# z = y**3 - y
# z = y * (1 + x**2)

y = np.array([
    x[0] ** 2,
    x[0],
])

fig = plt.figure('xy')
fig.clf()
ax = fig.subplots(1, 1)

ax.hist2d(y[0], y[1], bins=int(x.shape[1] ** 0.25), norm=colors.LogNorm())

fig.show()
