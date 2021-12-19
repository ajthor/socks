Systems
=======

Main Points
-----------

Custom dynamical systems should inherit from
:py:class:`~gym_socks.envs.dynamical_system.DynamicalSystem`. This is the main interface
for dynamical system environments, and differs mainly from ``gym.Env`` in that it adds a
:py:attr:`~gym_socks.envs.dynamical_system.DynamicalSystem.state_space` attribute and
defines a custom :py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.step`
function that solves an initial value problem to compute the subsequent state of the
system and calls
:py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.generate_disturbance` and
:py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.generate_observation`
internally.

Typically, the main methods to be overridden in custom systems are
:py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.dynamics` and
:py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.generate_disturbance`.

.. important::

    Remember that :py:meth:`~gym_socks.envs.dynamical_system.DynamicalSystem.dynamics`
    is defined using the dynamical system equations in :eq:`dynamics`. This makes the
    code easier since in many cases the system dynamics are given in continuous time,
    but need to be simulated in discrete time.

Templates
---------

It is recommended to use the following templates when defining new systems
(environments) to use in algorithms.

Default Template
~~~~~~~~~~~~~~~~

The default template defines a dynamical system with dynamics given by:

.. math::
    :label: dynamics

    \dot{x} &= f(x, u, w) \\
    y &= h(x, u, v)

.. code-block:: python

    import gym
    from gym_socks.envs.dynamical_system import DynamicalSystem

    import numpy as np


    class CustomDynamicalSystemEnv(DynamicalSystem):

        def __init__(self, seed=0, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

            self.state = None

            self.seed(seed=seed)

        def generate_disturbance(self, time, state, action):
            w = self.np_random.standard_normal(size=self.state_space.shape)
            return 1e-2 * np.array(w)

        def generate_observation(self, time, state, action):
            v = self.np_random.standard_normal(size=self.observation_space.shape)
            return np.array(state, dtype=np.float32) + np.array(v)

        def dynamics(self, time, state, action, disturbance):
            ...

The system can then be "registered" using the OpenAI gym ``register`` function in order
to "make" the system via a string identifier. This is useful for configuring experiments
using `sacred <https://github.com/IDSIA/sacred>`_, and for ensuring correct versioning
of environments for repeatability.

.. code-block:: python

    from gym.envs.registration import register

    register(
        id="CustomDynamicalSystemEnv-v0",
        entry_point="CustomDynamicalSystemEnv",
        order_enforce=False,
    )

.. note::

    The ``step``, ``reset``, ``render``, and ``close`` methods are inherited from
    ``gym.Env`` and should be overridden if custom behavior is needed, for instance if
    explicitly using linear dynamics :math:`x_{t+1} = A x_{t} + B u_{t} + w_{t}` is
    desired.

Discrete Time Linear System Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This template defines a dynamical system with dynamics given by:

.. math::

    x_{t+1} = A x_{t} + B u_{t} + w_{t}

.. code-block:: python

    import gym
    from gym_socks.envs.dynamical_system import DynamicalSystem

    import numpy as np


    class CustomDynamicalSystemEnv(DynamicalSystem):

        def __init__(self, seed=0, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

            self.state = None

            self.state_matrix = np.zeros(shape=(2, 2))  # <-- change this
            self.input_matrix = np.zeros(shape=(2, 1))  # <-- change this

            self.seed(seed=seed)

        def step(self, time, action):

            disturbance = self.generate_disturbance(time, self.state, action)
            self.state = self.dynamics(time, self.state, action, disturbance)
            obs = self.generate_observation(time, self.state, action)

            return obs, 0, False, {}

        def generate_disturbance(self, time, state, action):
            w = self.np_random.standard_normal(size=self.state_space.shape)
            return 1e-2 * np.array(w)

        def dynamics(self, time, state, action, disturbance):
            return self.state_matrix @ state + self.input_matrix @ action + disturbance