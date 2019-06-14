import numpy as np
from matplotlib import pyplot as plt

N = 10000

x = np.random.randn(N) + 5
y = np.random.randn(N)

# z = np.sign(y) * np.log(1 + np.abs(y))
# z = y**3 - y
z = y * (1 + x**2)

fig = plt.figure('xy')
fig.clf()
ax = fig.subplots(1, 1)

ax.hist(z, bins='auto', histtype='step')

fig.show()
