:py:mod:`~gym_socks.envs.sample`
================================

.. py:module:: gym_socks.envs.sample

.. autoapi-nested-parse::

   Sampling methods.

   This file contains a collection of sampling methods. The core principle is to define a
   function that returns a single observation (either via return or yield) from a
   probability measure. Then, the `sample_generator` is a decorator, which converts a
   function that returns a single observation into a generator, that can be sampled using
   `islice`.

   .. rubric:: Example

   Sample the stochastic kernel of a dynamical system (i.e. the state transition
   probability kernel).

       >>> env = NdIntegrator(2)
       >>> sample_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
       >>> sampler = step_sampler(
       ...     system=env, policy=RandomizedPolicy(env), sample_space=sample_space
       ... )
       >>> S = sample(sampler=sampler, sample_size=100)

   The main reason for this setup is to allow for 'observation functions' which have
   different structures, e.g.
   * a function which `return`s an observation,
   * an infinite generator which `yield`s observations, and
   * a finite generator that `yield`s observations.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.envs.sample.sample_generator
   gym_socks.envs.sample.step_sampler
   gym_socks.envs.sample.uniform_grid
   gym_socks.envs.sample.uniform_grid_step_sampler
   gym_socks.envs.sample.sequential_action_sampler
   gym_socks.envs.sample.trajectory_sampler
   gym_socks.envs.sample.sample
   gym_socks.envs.sample.transpose_sample
   gym_socks.envs.sample.reshape_trajectory_sample



.. py:function:: sample_generator(fun)

   Sample generator decorator.

   Converts a sample function into a generator function. Any function that returns a
   single observation (as a tuple) can be converted into a sample generator.

   :param fun: Sample function that returns or yields an observation.

   :returns: A function that can be used to `islice` a sample from the sample generator.

   .. rubric:: Example

   >>> from itertools import islice
   >>> from gym_socks.envs.sample import sample_generator
   >>> @sample_generator
   ... def custom_sampler(system, policy, sample_space):
   ...     system.state = sample_space.sample()
   ...     action = policy(state=state)
   ...     next_state, *_ = system.step(action)
   ...     yield (system.state, action, next_state)
   >>> S = list(islice(custom_sampler(), 100))


.. py:function:: step_sampler(env = None, policy = None, sample_space = None)

   Default sampler (one step).

   :param env: The system to sample from.
   :param policy: The policy applied to the system during sampling.
   :param sample_space: The space where initial conditions are drawn from.

   :returns: A generator function that yields system observations as tuples.


.. py:function:: uniform_grid(xi)

   Create a uniform grid from a list of ranges.

   :param xi: List of ranges.

   :returns: Grid of points (the product of all points in ranges).

   .. rubric:: Example

   >>> import numpy as np
   >>> from gym_socks.envs.sample import uniform_grid
   >>> grid = uniform_grid([np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)])


.. py:function:: uniform_grid_step_sampler(xi, env = None, policy = None, sample_space = None)

   Uniform sampler (one step).

   :param xi: List of ranges.
   :param env: The system to sample from.
   :param policy: The policy applied to the system during sampling.
   :param sample_space: The space where initial conditions are drawn from.

   :returns: A generator function that yields system observations as tuples.


.. py:function:: sequential_action_sampler(ui, env, sample_space, sampler)


.. py:function:: trajectory_sampler(time_horizon, env = None, policy = None, sample_space = None)

   Default trajectory sampler.

   :param env: The system to sample from.
   :param policy: The policy applied to the system during sampling.
   :param sample_space: The space where initial conditions are drawn from.

   :returns: A generator function that yields system observations as tuples.


.. py:function:: sample(sampler=None, sample_size = None, *args, **kwargs)

   Generate a sample using the sample generator.

   :param sampler: Sample generator function.
   :param sample_size: Size of the sample.

   :returns: list of tuples


.. py:function:: transpose_sample(sample)

   Transpose the sample.

   By default, a sample should be a list of tuples of the form::

       S = [(x_1, y_1), ..., (x_n, y_n)]

   For most algorithms, we need to isolate the sample components (e.g. all x's).
   This function converts a sample from a list of tuples to a tuple of lists::

       S_T = ([x_1, ..., x_n], [y_1, ..., y_n])

   This can then be unpacked as: ``X, Y = S_T``

   :param sample: list of tuples

   :returns: tuple of lists


.. py:function:: reshape_trajectory_sample(sample)

   Reshapes trajectory samples.

   Often, trajectory samples are organized such that the "trajectory" components are a
   2D array of points indexed by time. However, for kernel methods, we typically
   require that the trajectories be concatenated into a single vector (1D array)::

       [[x1], [x2], ..., [xn]] -> [x1, x2, ..., xn]

   This function converts the sample so that the trajectories are 1D arrays.

   :param sample: list of tuples

   :returns: List of tuples, where the components of the tuples are flattened.
