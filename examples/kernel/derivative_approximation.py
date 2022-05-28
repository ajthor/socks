# %% [markdown]
"""
# Derivative Approximation

This example demonstrates the use of conditional distribution embeddings to compute an
approximation of the first and second derivatives of a function,

$$f(x) = x^{3} - 4 x^{2} + 6 x - 24 + \exp(-x)$$

We use an idealized dataset, which approximates the partial derivatives well.
Introducing noise to the dataset creates fluctuations that needs to be offset by a
higher regularization parameter and smoother kernel.

To run the example, use the following command:

```shell
    python examples/kernel/derivative_approximation.py
```

"""

# %%
import numpy as np
from functools import partial

from gym_socks.kernel.metrics import rbf_kernel
from gym_socks.kernel.metrics import rbf_kernel_derivative
from gym_socks.kernel.metrics import regularized_inverse

from time import perf_counter

import matplotlib
from matplotlib import pyplot as plt

# %% [markdown]
# ## Generate the Sample
#
# The following sample is idealized in order to demonstrate the technique. Additive
# noise causes fluctuations in the derivative approximation.

# %%
sample_size = 200
X_train = np.linspace(-5, 5, sample_size).reshape(-1, 1)
y_train = X_train ** 3 - 4 * X_train ** 2 + 6 * X_train - 24 + np.exp(-X_train)

# Uncomment the following line to incorporate noise in the sample.
# y_train += 10 * np.random.standard_normal(size=(sample_size, 1))  # Additive noise.

X_test = np.linspace(-4, 4, 1000).reshape(-1, 1)

# %% [markdown]
# ## Kernel and Parameters

# %%
sigma = 0.1
kernel_fn = partial(rbf_kernel, sigma=sigma)
regularization_param = 1 / (sample_size ** 2)

# %% [markdown]
# ## Compute the Approximation

#  %%
start = perf_counter()

G = kernel_fn(X_train)
K = kernel_fn(X_train, X_test)

W = regularized_inverse(G, regularization_param=regularization_param)

C = rbf_kernel_derivative(X_train, sigma=sigma)
D = rbf_kernel_derivative(X_train, X_test, sigma=sigma)

y_pred = y_train.T @ W @ K
y_pred_d1 = -y_train.T @ W @ (K * D)
y_pred_d2 = (y_train.T @ W @ (G * C)) @ W @ (K * D)

print(perf_counter() - start)

# %% [markdown]
# ## Plot the Results
#
# We then plot the original function, the first derivative, and the second derivative
# against the original data.

# %%
fig = plt.figure()
ax = plt.axes()
plt.grid()

plt.scatter(X_train, y_train, marker=".", c="grey", label="Data")
plt.plot(X_test, y_pred.T, linewidth=2, label="Function Approximation")
plt.plot(X_test, y_pred_d1.T, linewidth=2, label="First Derivative")
plt.plot(X_test, y_pred_d2.T, linewidth=2, label="Second Derivative")

plt.legend()

plt.show()
