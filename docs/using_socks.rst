Using SOCKS
===========

Defining Custom Systems
-----------------------

Main Points
~~~~~~~~~~~

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
    is defined using dynamics:

    .. math::

        \dot{x} = f(x, u, w)

    This makes the code easier since in many cases the system dynamics are given in
    continuous time, but need to be simulated in discrete time.

System Templates
~~~~~~~~~~~~~~~~

It is recommended to use the following templates when defining new systems
(environments) to use in algorithms.

.. tab-set::

    .. tab-item:: Default

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

                def dynamics(self, time, state, action, disturbance):
                    ...


        The default template defines a stochastic dynamical system with dynamics given
        by:

        .. math::

            \dot{x} = f(x, u, w)

    .. tab-item:: Discrete-Time Linear

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

                    self.A = np.zeros(shape=(2, 2))  # <-- change this
                    self.B = np.zeros(shape=(2, 1))  # <-- change this

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
                    return self.A @ state + self.B @ action + disturbance


        A discrete-time linear system has dynamics given by:

        .. math::

            x_{t+1} = A x_{t} + B u_{t} + w_{t}


    .. tab-item:: Partially Observable

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
                    v = self.np_random.standard_normal(
                        size=self.observation_space.shape
                    )
                    return np.array(state, dtype=np.float32) + np.array(v)

                def dynamics(self, time, state, action, disturbance):
                    ...


        A partially observable system includes an observation function :math:`h`. It is
        usually used when the state observations are corrupted by some sort of noise
        process.

        .. math::

            \dot{x} &= f(x, u, w) \\
            y &= h(x, u, v)

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

Defining Custom Policies
------------------------

By definition, a policy is a function which returns a control action. In SOCKS, a policy
is defined as a callable class that inherits from
:py:class:`~gym_socks.envs.policy.BasePolicy`, which is also used as a parent class for
algorithms which compute control policies.

Thus, a policy simply inherits from :py:class:`~gym_socks.envs.policy.BasePolicy` and
implements a custom :py:meth:`~gym_socks.envs.policy.BasePolicy.__call__` function.

.. important::

    The main thing to know when defining custom policies is that functions in SOCKS
    which sample from systems pass all relevant information to the policy for the
    purpose of computing a control action. Since the policy can be either time-varying
    or time-invariant or open- or closed-lopp, it may require the time and state of the
    system in order to compute the control action. Thus, custom policies should
    explicitly name these variables as parameters if they are required, and should use
    ``*args, **kwargs`` to capture additional, unneeded arguments.

Policy Templates
~~~~~~~~~~~~~~~~

It is recommended to use the following template when defining new policies to use in
algorithms.

.. code-block:: python

    from gym_socks.envs.policy import BasePolicy

    class CustomPolicy(BasePolicy):

        def __init__(self, action_space: Space = None):
            self.action_space = action_space

        def __call__(self, *args, **kwargs):
            return self.action_space.sample()


Sampling Functions
------------------

A sample (otherwise known as a dataset) is a sequence of observations taken from a
stochastic kernel. Given system dynamics, a system observation consists of the starting
system state, an applied control action, and the resulting state.

.. math::

    x_{t+1} = f(x_{t}, u_{t}, w_{t})

A sample :math:`\mathcal{S}` of size :math:`n \in \mathbb{N}` is a collection of
observations.

.. math::

    \mathcal{S} = \lbrace (x_{i}, u_{i}, x_{i}{}^{\prime}) \rbrace_{i=1}^{n}

where :math:`x_{i}` and :math:`u_{i}` are sampled i.i.d. from distributions on the state
space :math:`\mathcal{X}` and control space :math:`\mathcal{U}`, and
:math:`x_{i}{}^{\prime}` is a state in :math:`\mathcal{X}` that denotes the
state at the next time step, computed using the dynamics :math:`f`.


Main Points
~~~~~~~~~~~

The main idea of sampling using SOCKS is to define a generator function that yields a
single observation in the sample :math:`\mathcal{S}`. An infinite generator can be used
directly, since it can be used to generate a sample of any length. However, not all
generator functions are easily defined as infinite generators. A finite generator (or
even a non-generator function that returns an observation) can still be used, and SOCKS
provides a decorator called :py:func:`sample_generator` that wraps a sample function,
effectively turning it into an infinite generator.

The important points to remember:

* The states :math:`x_{i}` are drawn from a distribution on :math:`\mathcal{X}`.
* The actions :math:`u_{i}` are drawn from a distribution on :math:`\mathcal{U}`.
* The resulting states :math:`x_{i}{}^{\prime}` are drawn from a stochastic kernel.

Thus, we define a sample generator for states and actions, and another to generate the
tuple :math:`(x_{i}, u_{i}, x_{i}{}^{\prime})`. Very simply, a sample generator for
actions could be implemented as :py:obj:`action_space.sample()`, which is implemented in
:py:obj:`gym`.

.. note::

    A stochastic kernel is the mathematical way to define a conditional distribution. In
    RL, this is commonly known as a transition kernel, or a state transition function.

Sampling Function Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Basic

        .. code-block:: python

            state_sampler = random_sampler(sample_space=env.state_space)
            action_sampler = random_sampler(sample_space=env.action_space)

            @sample_generator
            def custom_sampler():
                state = next(state_sampler)
                action = next(action_sampler)

                env.state = state
                next_state, *_ = env.step(action=action)

                yield (state, action, next_state)

    .. tab-item:: Policy

        .. code-block:: python

            state_sampler = random_sampler(sample_space=env.state_space)
            policy = ConstantPolicy(action_space=env.action_space, constant=0)

            @sample_generator
            def custom_sampler():
                state = next(state_sampler)
                action = policy()

                env.state = state
                next_state, *_ = env.step(action=action)

                yield (state, action, next_state)

Making Experiments
------------------

We strongly encourage using the `sacred <https://github.com/IDSIA/sacred>`_ experimental
framework for new experiments. This encourages repeatability and easy configuration that
is useful for rapid development, iteration, and modification.

Minimal Experiment
~~~~~~~~~~~~~~~~~~

The following template is a (mostly) minimal example of how to organize a new
experiment, and allows for simple modification and extension.

.. code-block:: python

    import gym
    import gym_socks
    import logging

    import numpy as np

    from sacred import Experiment

    from examples._computation_timer import ComputationTimer

    from examples.ingredients.system_ingredient import system_ingredient
    from examples.ingredients.system_ingredient import set_system_seed
    from examples.ingredients.system_ingredient import make_system


    @system_ingredient.config
    def system_config():
        system_id = "2DIntegratorEnv-v0"
        sampling_time = 0.25


    ex = Experiment(ingredients=[system_ingredient])


    @ex.config
    def config():
        time_horizon = 5
        results_filename = "results/data.npy"


    @ex.main
    def main(seed, time_horizon, results_filename, _log):

        env = make_system()

        # Set the random seed.
        set_system_seed(seed, env)

        with ComputationTimer():

            # Run algorithm here.

        # Save the result to NPY file.
        _log.debug(f"Saving the results to file {results_filename}.")
        with open(results_filename, "wb") as f:
            np.save(f, result)


    if __name__ == "__main__":
        ex.run_commandline()

The "magic" of sacred comes from the configuration functions, and how the variables
defined within them can be updated and injected into the ``main`` function. However, the
main ideas are to define parameters and configuration variables within the ``config``
function, and to add them as arguments to the ``main`` function. Then, at runtime,
sacred will pass either the default value specified in the config, or use the updated
values passed by the user at the command line. In fact, this works for any "captured"
function, which means that only the relevant configuration variables need to be passed
to the functions that use them. This allows for more modularized experiments and also
allows for more convenient interaction by the user.

Adding Plotting
~~~~~~~~~~~~~~~

Plotting results is typically desired for many experiments, and the sacred framework can
also be used to create configurable plotting commands.

The following code can be added to the experiment, and then a plot can be generated
using the command:

.. code-block:: shell

    python -m <experiment> plot_results

.. code-block:: python

    from examples.ingredients.plotting_ingredient import plotting_ingredient
    from examples.ingredients.plotting_ingredient import update_rc_params

    @plotting_ingredient.config_hook
    def _plot_config(config, command_name, logger):
        if command_name in {"main", "plot_results"}:
            return {
                "axes": {
                    "xlabel": r"$x_1$",
                    "ylabel": r"$x_2$",
                }
            }


    @ex.command(unobserved=True)
    def plot_results(plot_cfg, results_filename):

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Dynamically load for speed.
        import matplotlib

        matplotlib.use("Agg")
        update_rc_params(matplotlib, plot_cfg["rc_params"])

        import matplotlib.pyplot as plt

        # Load the result from NPY file.
        with open(results_filename, "rb") as f:
            result = np.load(f)

        fig = plt.figure()
        ax = plt.axes(**plot_cfg["axes"])

        # Plotting code here.

        plt.savefig(plot_cfg["plot_filename"])

The utility of this approach is when the format of the results needs to be changed to
fit a particular publication, but re-running the experiment can be time-intensive. By
saving the main algorithm results to a file, and then loading them separately in the
plotting function, we save the time of having to re-compute the algorithm for plot
manipulation. The :py:mod:`~examples.ingredients.plotting_ingredient` module implements
a small ingredient that can be used to modify the ``rc_params`` of ``matplotlib``, or
add configuration options that can be used to modify the plot appearance. For example,

.. code-block:: shell

    python -m <experiment> plot_results with <updates>

.. note::

    The plotting config is typically hidden when using the ``print_config`` command
    using the above code. This is due to the ``if command_name in {"main",
    "plot_results"}`` line above. To show the config, either use the ``-p`` flag when
    calling the ``plot_results`` command, or remove the ``if`` statement in the
    ``_plot_config`` function above.