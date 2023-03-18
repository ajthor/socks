***********
Using SOCKS
***********

.. toctree::
    :hidden:
    :maxdepth: 1

    for_gym_users


In practice, we are typically given a dataset to work with, either from an actual system
or from historical observations of a system. However, as researchers, we typically need
to generate a sample via simulation in order to develop and test algorithms.

In the following sections, we describe how to organize data that is given to us from a
pre-existing dataset and how to generate synthetic data by simulating a system.

Organizing Data
===============

If a dataset is already available, we need to organize and format the data correctly to
work with the algorithms in SOCKS. In this section, we will describe how to format and
organize the data into the correct format, and give some general cases.

Data Arrays
-----------

In a dynamical system, we generally observe transitions from the dynamical system
equations, meaning our data consists of states :math:`x_{i}`, the applied control
actions :math:`u_{i}`, and the resulting state after a single time step :math:`y_{i}`.

In python, data is organized in a 2D array with **one data point per row**. We want to
organize the states into an array ``X``, the control actions into an array ``U``, and
the resulting states into an array ``Y``. If we have states that are
:math:`n`-dimensional and :math:`M` sample points, then the arrays ``X`` and ``Y`` will
have dimensions :math:`M \times n`. Similarly, if the control actions are
:math:`m`-dimensional, then the array ``U`` will have dimensions :math:`M \times m`.

.. caution::

    Note that this is different from how data is typically ordered in Matlab, where it
    is easier to order data in *columns* due to the way Matlab orders the elements of
    matrices. Be careful when importing data from Matlab, since it may need to be
    transposed in order to fit the correct format.

.. seealso::

    If you have data available in Matlab, and you need to import it into python, check
    out `scipy.io.loadmat`_.

.. _scipy.io.loadmat:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

.. tip::

    If we are dealing with an uncontrolled system, meaning there are no control actions,
    specify ``U`` as an array of zeros, i.e. ``U = np.zeros(M, 1)``, where ``M`` is the
    number of sample points.

Mathematically, we typically organize a sample (dataset) as a collection of tuples,

.. math::

    \mathcal{S} = \lbrace (x_{i}, u_{i}, y_{i}) \rbrace_{i=1}^{M}

SOCKS uses the mathematical organization, meaning if we construct a set of 2D arrays, we
need to convert this to a list of tuples. Luckily, this is a simple operation in python,
and we can use the built-in ``zip`` function.

Common Cases
------------

Here we outline some common cases, and give a small code snippet describing how to
collect and organize the data.

State Transitions
^^^^^^^^^^^^^^^^^

The most general case is where data is given as state transitions. We have the initial
states :math:`x_{i}`, the control actions applied to the system :math:`u_{i}`, and the
next states of the system after a single time step :math:`y_{i}`. The first step in
organizing the data should be to load all of the data into arrays ``X``, ``U``, and
``Y``. This part is up to you. Once you have these arrays, we need to convert this to a
sample :math:`\mathcal{S}`.

If ``X``, ``U``, and ``Y`` are correctly organized, such that each row in the arrays
corresponds to a single data point, then converting to a sample is straightforward using
the ``zip`` function.

.. code-block:: python

    S = list(zip(X, U, Y))  # A list of tuples: [(x1, u1, y1), ..., (xn, un, yn)].

Trajectory Data
^^^^^^^^^^^^^^^

Sometimes, dynamical system data is given as trajectories (meaning a sequence of states
and control actions indexed by time), and we would like to convert this to a collection
of state transitions. In this case, we need to break the trajectory down into its
"increments".

Regardless of how you choose to load the trajectories into python, we typically have a
2D array of states that are indexed by time. I.e. the trajectory has dimensions :math:`N
\times n`, where :math:`N` is the number of time steps and :math:`n` is the
dimensionality of the state space. In addition, we may have a sequence of control
actions applied at each time step (excluding the last time step), which has dimensions
:math:`N-1 \times m`, where :math:`m` is the dimensionality of the control or input
space. In order to split this into increments, we just need to slice the array
appropriately.

Suppose we are given a state ``trajectory`` as a 2D array. Then in order to separate it
into the correct format, we can use:

.. code-block:: python

    X = trajectory[:-1]  # Get all rows except the last.
    Y = trajectory[1:]  # Get all rows except the first.

Then, we can combine these into the correct format for socks using the ``zip`` function
as described above.

Note that if we have multiple trajectories, we then need to concatenate the states from
each trajectory, e.g. using `numpy.vstack`_.

.. _numpy.vstack: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html

.. important::

    When splitting a trajectory like this, it is important to make sure that the rows of
    ``X`` line up with the corresponding control actions in ``U`` and the resulting
    states in ``Y``. Be careful when concatenating arrays and slicing them to ensure
    that the state transitions line up correctly.

Simulating Systems
==================

In some cases, we may not be dealing with a pre-existing dataset. In this case, we
typically need to generate an artificial dataset by simulating a system.

SOCKS uses the OpenAI gym framework to simulate systems (called "environments" in gym).
For example, the following code can be used to simulate a 2D stochastic integrator
system over 10 time steps.

.. code-block:: python

    from gym_socks.envs.integrator import NDIntegrator
    env = NDIntegrator(dim=2)  # A 2D stochastic integrator system.

    # Reset the system to a random initial condition.
    env.reset()

    for t in range(10):
        action = env.action_space.sample()  # Generate a random control action.
        obs, *_ = env.step(time=t, action=action)

The :py:meth:`reset` function resets the system to a random initial condition. Then,
inside the ``for`` loop, we compute a control action, feed it to the dynamics using the
:py:meth:`step` function, and obtain an observation of the system state ``obs``.

A full list of the environments included in SOCKS is provided in the
:doc:`/api/gym_socks/envs/index` section of the :doc:`/api/index`.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - System
      - Identifier
      - Type
      - State Dim
      - Input Dim
    * - :py:class:`~gym_socks.envs.cwh.CWH4DEnv`
      - ``CWH4DEnv-v0``
      - Linear
      - 4
      - 2
    * - :py:class:`~gym_socks.envs.cwh.CWH6DEnv`
      - ``CWH6DEnv-v0``
      - Linear
      - 6
      - 3
    * - :py:class:`~gym_socks.envs.integrator.NDIntegratorEnv`
      - ``2DIntegratorEnv-v0``
      - Linear
      - N [#ND]_
      - 1
    * - :py:class:`~gym_socks.envs.nonholonomic.NonholonomicVehicleEnv`
      - ``NonholonomicVehicleEnv-v0``
      - Nonlinear
      - 3
      - 2
    * - :py:class:`~gym_socks.envs.planar_quad.PlanarQuadrotorEnv`
      - ``PlanarQuadrotorEnv-v0``
      - Nonlinear
      - 6
      - 2
    * - :py:class:`~gym_socks.envs.point_mass.NDPointMassEnv`
      - ``2DPointMassEnv-v0``
      - Linear
      - N [#ND]_
      - N [#ND]_


.. [#ND] The string identifier for these systems generates a 2D system.

Generating a Sample
-------------------

The main idea of sampling using SOCKS is to define a generator function that yields a
tuple in the sample :math:`\mathcal{S}`,

.. math::

    \mathcal{S} = \lbrace (x_{i}, u_{i}, y_{i}) \rbrace_{i=1}^{M}

.. note::

    A generator in python can be thought of as a function that remembers its state and
    may be called multiple times to return a sequence of values. In our case, we use it
    to "generate" a sequence of observations.

.. tip::

    An infinite generator can be used directly, since it can be used to generate a
    sample of any length. However, not all generator functions are easily defined as
    infinite generators. A finite generator (or even a non-generator function that
    returns an observation) can still be used, and SOCKS provides a decorator called
    :py:func:`sample_fn` that wraps a sample function, effectively turning it
    into an infinite generator.

The important points to remember:

* The states :math:`x_{i}` are drawn from a distribution on :math:`\mathcal{X}`.
* The actions :math:`u_{i}` are drawn from a distribution on :math:`\mathcal{U}`.
* The resulting states :math:`y_{i}` are drawn from a stochastic kernel.

In order to generate a sample, we need a way to generate random states, compute control
actions (which can either be chosen randomly or chosen via a policy), and compute the
resulting states. SOCKS implements several functions to make this process easier. In a
nutshell, we need to define a sample generator for states and actions, and another to
generate the tuple :math:`(x_{i}, u_{i}, y_{i})`.

In order to sample from a space, we can use the
:py:func:`~gym_socks.sampling.sample.space_sampler` function, which randomly generates
samples from a :py:obj:`Space`.

.. code-block:: python

    from gym_socks.envs.spaces import Box
    from gym_socks.sampling import space_sampler

    sample_space = Box(low=-1, high=1, shape(2,), dtype=float)

    # Generate a single observation from the space.
    x = next(space_sampler)

    # Generate a sample of 100 observations.
    X = space_sampler(space=sample_space).sample(100)

.. seealso::

    SOCKS also provides a sampling function which generates points from a pre-specified
    grid, :py:func:`~gym_socks.sampling.sample.grid_sampler`. This can be useful in
    certain cases to obtain a more uniform result.

We can use the same procedure to randomly sample from the state and action spaces of an
environment.

.. code-block:: python

    from gym_socks.envs.integrator import NDIntegratorEnv
    from gym_socks.sampling import space_sampler

    env = NDIntegratorEnv(2)
    state_sampler = space_sampler(env.state_space)
    action_sampler = space_sampler(env.action_space)

Then, SOCKS implements a function
:py:func:`~gym_socks.sampling.sample.transition_sampler`, which can be used to generate
the tuples in :math:`\mathcal{S}`. Continuing from the last code block,

.. code-block:: python

    from gym_socks.sampling import transition_sampler

    # Generate a sample of 100 observations.
    S = transition_sampler(env, state_sampler, action_sampler).sample(100)

.. seealso::

    There is also a sampling function,
    :py:func:`~gym_socks.sampling.sample.trajectory_sampler`, which generates samples
    consisting of initial conditions, control sequences, and resulting trajectories.
    This can be useful depending on the algorithm you are using.

Custom Sampling Functions
-------------------------

Sometimes, the built-in sampling functions in SOCKS may be insufficient for your needs.
If you need to manipulate the way the samples are generated, for instance to change the
sampling distribution, then you may need to define your own sampling function. This is
achievable in SOCKS using the :py:func:`~gym_socks.sampling.sample.sample_fn` decorator.

.. tab-set::

    .. tab-item:: Basic

        .. code-block:: python

            from gym_socks.sampling import sample_fn
            from gym_socks.sampling import space_sampler

            from gym_socks.envs.integrator import NDIntegrator

            env = NDIntegrator(dim=2)  # A 2D stochastic integrator system.

            state_sampler = space_sampler(space=env.state_space)
            action_sampler = space_sampler(space=env.action_space)

            @sample_fn
            def custom_sampler():
                state = next(state_sampler)
                action = next(action_sampler)

                env.reset(state)
                next_state, *_ = env.step(action=action)

                return state, action, next_state

            # Generate a sample of 100 observations.
            S = custom_sampler().sample(size=100)

    .. tab-item:: Policy

        .. code-block:: python
            :emphasize-lines: 5, 10, 15

            from gym_socks.sampling import sample_fn
            from gym_socks.sampling import space_sampler

            from gym_socks.envs.integrator import NDIntegrator
            from gym_socks.policies import RandomizedPolicy

            env = NDIntegrator(dim=2)  # A 2D stochastic integrator system.

            state_sampler = space_sampler(space=env.state_space)
            policy = RandomizedPolicy(action_space=env.action_space)

            @sample_fn
            def custom_sampler():
                state = next(state_sampler)
                action = policy(state=state)

                env.reset(state)
                next_state, *_ = env.step(action=action)

                return state, action, next_state

            # Generate a sample of 100 observations.
            S = custom_sampler().sample(size=100)

Check out the :doc:`/guide/templates` page for some templates and a description of how
you can create your own sampling functions.

Algorithms
==========

The algorithms in SOCKS are mostly designed to follow the format of `scikit-learn`_,
meaning they typically have some kind of "fit/predict" function to train the algorithm
and then evaluate the solution.

.. _scikit-learn: https://scikit-learn.org/stable/index.html

Because the algorithms handle a number of different and distinct problems, we do not
describe them here. Instead, check out the :doc:`/examples/index` page to get an idea of
how they work and how to use them.

If you have questions or get stuck, feel free to reach out by posting an `issue
<https://github.com/ajthor/socks/issues>`_ on GitHub or by starting a `discussion
<https://github.com/ajthor/socks/discussions>`_.

We'd love to hear from you!