import matplotlib.pyplot as plt
import numpy as np

import build_functions
import spectral_analysis

mu, sigma = 1.0, 0.5  # mean and standard deviation
s = np.random.normal(mu, sigma, 20000)

plt.ion()
plt.figure()
count, bins, ignored = plt.hist(s, 30, density=True)

plt.plot(
    bins,
    1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
    linewidth=2,
    color="r",
)
plt.title("Mu")
plt.tight_layout()
plt.show()


mu, sigma = 2, 0.5  # mean and standard deviation
s = np.random.normal(mu, sigma, 20000)

plt.ion()
plt.figure()
count, bins, ignored = plt.hist(s, 30, density=True)

plt.plot(
    bins,
    1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
    linewidth=2,
    color="r",
)
plt.title("Kx")

plt.tight_layout()
plt.show()
