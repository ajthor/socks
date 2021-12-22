# %% [markdown]
"""
# Forward Reachability

This example demonstrates the forward reachability classifier on a set of dummy data.
Note that the data is not taken from a dynamical system, but can easily be adapted to
data taken from system observations via a simple substitution. The reason for the dummy
data is to showcase the technique on a non-convex forward reachable set.

To run the example, use the following command:

```shell
    python examples/reach/forward_reach.py
```

"""

# %%
import gym_socks

import numpy as np

from gym.envs.registration import make

from gym_socks.algorithms.reach.separating_kernel import SeparatingKernelClassifier

from functools import partial
from sklearn.metrics.pairwise import euclidean_distances

from gym_socks.utils.grid import make_grid_from_ranges

from gym_socks.policies import ConstantPolicy
from gym_socks.sampling import sample
from gym_socks.sampling import sample_generator
from gym_socks.sampling import random_sampler

# %% [markdown]
# ## Generate The Sample
#
# We demonstrate the use of the algorithm on a non-convex region. We choose to sample
# uniformly within a toroidal region centered around the origin.

# %%
sigma = 0.1
sample_size = 1000

regularization_param = 1 / sample_size


@sample_generator
def sampler() -> tuple:
    """Sample generator.

    Sample generator that generates points in a donut-shaped ring around the origin.
    An example of a non-convex region.

    Yields:
        sample : A sample taken iid from the region.

    """

    r = np.random.uniform(low=0.5, high=0.75, size=(1,))
    phi = np.random.uniform(low=0, high=2 * np.pi, size=(1,))
    point = np.array([r * np.cos(phi), r * np.sin(phi)])

    yield tuple(np.ravel(point))


# Sample the distribution.
S = sample(sampler=sampler, sample_size=sample_size)

# %% [markdown]
# ## Algorithm
#
# We then run the algorithm to compute the classification boundary. This can be
# evaluated easily for a large number of test points.

# %%
# Construct the algorithm.
alg = SeparatingKernelClassifier(
    kernel_fn=partial(
        gym_socks.kernel.metrics.abel_kernel,
        sigma=sigma,
        distance_fn=euclidean_distances,
    ),
    regularization_param=regularization_param,
)

# Generate test points.
T = make_grid_from_ranges([np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)])

# Train the classifier and classify the points.
labels = alg.fit(S).predict(T)

# %% [markdown]
# ## Results
#
# We plot the results here using "pixels" to represent the points predicted to be within
# the classification boundary. Since the classifier is point-based, it is difficult to
# plot the "set" defined by the algorithm. Since the set is non-convex, we cannot use a
# countour plot that relies upon a convex hull.

# %%
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes()

points_in = T[labels == True]

size = (300.0 / fig.dpi) ** 2
plt.scatter(points_in[:, 0], points_in[:, 1], color="C0", marker=",", s=size)
S = np.array(S)
plt.scatter(S[:, 0], S[:, 1], color="r", marker=".", s=1)

# Plot support region.
plt.gca().add_patch(plt.Circle((0, 0), 0.5, fc="none", ec="blue"))
plt.gca().add_patch(plt.Circle((0, 0), 0.75, fc="none", ec="blue"))

plt.show()

# %% [markdown]
# Cite as:
#
# ```bibtex
# @inproceedings{thorpe2021learning,
#   title     = {Learning Approximate Forward Reachable Sets Using Separating Kernels},
#   author    = {Thorpe, Adam J. and Ortiz, Kendric R. and Oishi, Meeko M. K.},
#   booktitle = {Proceedings of the 3rd Conference on Learning for Dynamics and Control},
#   pages     = {201--212},
#   year      = {2021},
#   volume    = {144},
#   series    = {Proceedings of Machine Learning Research},
#   month     = {07 -- 08 June},
#   publisher = {PMLR},
#   pdf       = {http://proceedings.mlr.press/v144/thorpe21a/thorpe21a.pdf},
#   url       = {https://proceedings.mlr.press/v144/thorpe21a.html}
# }
# ```
