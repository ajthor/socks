# %% [markdown]
"""
# Conditional Distribution Embedding

This example demonstrates the use of conditional distribution embeddings on a simple
function $x^{2}$, corrupted by Gaussian noise. This example is useful for
experimenting with the kernel choice, parameters, or dataset and visualizing the result.

To run the example, use the following command:

```shell
    python examples/kernel/conditional_embedding.py
```

"""

# %%
import numpy as np
from functools import partial
from gym_socks.algorithms.kernel import ConditionalEmbedding
from gym_socks.algorithms.kernel import GenerativeModel
from gym_socks.kernel.metrics import rbf_kernel

from time import perf_counter

import matplotlib
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor

# %% [markdown]
# ## Generate the Sample

# %%
sample_size = 20
X_train = np.random.uniform(-5.5, 5.5, sample_size).reshape(-1, 1)
# y_train = np.empty_like(X_train)
# y_train[:sample_size] = X_train[:sample_size] ** 2
# y_train[sample_size:] = -X_train[sample_size:] ** 2
y_train = X_train ** 2
y_train += 2 * np.random.standard_normal(size=(sample_size, 1))  # Additive noise.

X_test = np.linspace(-5, 5, 1000).reshape(-1, 1)

# %% [markdown]
# ## Kernel and Parameters
#
# - The regularization parameter affects the "smoothness" of the approximation. If the
#   value is too low, this will lead to overfitting. If the value is high, the
#   approximation may be overly smooth.
# - For the Gaussian (RBF) kernel, `sigma` controls the "bandwidth" of the Gaussian.
#   Decreasing this will allow the approximation to detect sharper changes in the
#   function, while a larger value will give a smoother approximation.

# %%
sigma = 5
kernel_fn = partial(rbf_kernel, sigma=sigma)
# regularization_param = 1 / (sample_size ** 2)
regularization_param = 1e-2

# %% [markdown]
# ## Compute the Appromation
#
# We now compute the estimated $y$ values of the function at new (unseen) values of
# $x$.

#  %%

fig = plt.figure()
ax = plt.axes()
plt.grid()

alg = ConditionalEmbedding(regularization_param=regularization_param)
y_mean = alg.fit(kernel_fn(X_train), y_train).predict(kernel_fn(X_train, X_test))

start = perf_counter()
alg = GenerativeModel(regularization_param=regularization_param)
alg.fit(kernel_fn(X_train), y_train)

# G = kernel_fn(X_test)
K = kernel_fn(X_train, X_test)
num_samples = 10
y_pred = alg.predict(K, num_samples)

print(perf_counter() - start)

plt.scatter(X_train, y_train, marker=".", c="grey", label="Data")
plt.plot(X_test, y_mean, linewidth=2, label="Empirical Mean")

for i in range(num_samples):
    plt.plot(X_test, y_pred[i], color="C0", alpha=0.5, linewidth=1)

plt.legend()

plt.savefig("results/plot.png")
