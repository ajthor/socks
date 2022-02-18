***********
Using SOCKS
***********

In practice, we are typically given a dataset to work with, either from an actual system
or from historical observations of a system. However, as researchers, we typically need
to generate a sample via simulation in order to develop and test algorithms.

Simulating Systems
==================

SOCKS uses the OpenAI gym framework to simulate systems (called environments in gym).
For example, the following code can be used to simulate a 2D stochastic integrator
system over 10 time steps.

.. code-block:: python

    import gym
    import gym_socks
    env = gym.make("2DIntegratorEnv-v0")
    env.reset()
    policy = RandomizedPolicy(action_space=env.action_space)
    for t in range(10):
        action = policy()
        obs, *_ = env.step(time=t, action=action)

The :py:meth:`reset` function resets the system to a random initial condition. We then
define a randomized policy, which selects a random control action from the action space.
This is equivalent to using ``env.action_space.sample()``, but we make explicit that the
policy is a separate object. Then, inside the ``for`` loop, we compute a control action,
feed it to the dynamics using the :py:meth:`step` function, and obtain an observation of
the system state ``obs``.

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


Sampling Functions
==================

The main idea of sampling using SOCKS is to define a generator function that yields a
single observation in the sample :math:`\mathcal{S}`.

.. note::

    A generator in Python can be thought of as a function that remembers its state and
    may be called multiple times to return a sequence of values. In our case, we use it
    to "generate" a sequence of observations.

.. hint::

    An infinite generator can be used directly, since it can be used to generate a
    sample of any length. However, not all generator functions are easily defined as
    infinite generators. A finite generator (or even a non-generator function that
    returns an observation) can still be used, and SOCKS provides a decorator called
    :py:func:`sample_generator` that wraps a sample function, effectively turning it
    into an infinite generator.

The important points to remember:

* The states :math:`x_{i}` are drawn from a distribution on :math:`\mathcal{X}`.
* The actions :math:`u_{i}` are drawn from a distribution on :math:`\mathcal{U}`.
* The resulting states :math:`x_{i}{}^{\prime}` are drawn from a stochastic kernel.

Thus, we define a sample generator for states and actions, and another to generate the
tuple :math:`(x_{i}, u_{i}, x_{i}{}^{\prime})`. Very simply, a sample generator for
actions could be implemented as :py:obj:`action_space.sample()`, which is implemented in
:py:obj:`gym`.

Sampling Function Examples
--------------------------

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
            policy = RandomizedPolicy(action_space=env.action_space)

            @sample_generator
            def custom_sampler():
                state = next(state_sampler)
                action = policy()

                env.state = state
                next_state, *_ = env.step(action=action)

                yield (state, action, next_state)


Making Experiments
==================

We strongly encourage using the `sacred <https://github.com/IDSIA/sacred>`_ experimental
framework for new experiments. This promotes repeatability and easy configuration that
is useful for rapid development, iteration, and modification.

Minimal Experiment
------------------

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
---------------

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

.. hint::

    The plotting config is typically hidden when using the ``print_config`` command
    using the above code. This is due to the ``if command_name in {"main",
    "plot_results"}`` line above. To show the config, either use the ``-p`` flag when
    calling the ``plot_results`` command, or remove the ``if`` statement in the
    ``_plot_config`` function above.