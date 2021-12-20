Experiments
===========

We strongly encourage using the `sacred <https://github.com/IDSIA/sacred>`_ experimental
framework for new experiments. This encourages repeatability and easy configuration that
is useful for rapid development, iteration, and modification.

Templates
---------

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