:py:mod:`examples.benchmark_obstacle_avoid_problem`
===================================================

.. py:module:: examples.benchmark_obstacle_avoid_problem

.. autoapi-nested-parse::

   Stochastic optimal control (obstacle avoid).

   This example demonstrates the optimal controller synthesis algorithm on an obstacle
   avoidance problemlem.

   By default, it uses a nonlinear dynamical system with nonholonomic vehicle dynamics.
   Other dynamical systems can also be used, by modifying the configuration as needed.

   Several configuration files are included in the `examples/configs` folder, and can be
   used by running the example using the `with` syntax, e.g.

       $ python -m <experiment> with examples/configs/<config_file>

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.benchmark_tracking_problem

   .. [1] `Stochastic Optimal Control via
           Hilbert Space Embeddings of Distributions, 2021
           Adam J. Thorpe, Meeko M. K. Oishi
           IEEE Conference on Decision and Control,
           <https://arxiv.org/abs/2103.12759>`_



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.benchmark_obstacle_avoid_problem.system_config
   examples.benchmark_obstacle_avoid_problem.sample_config
   examples.benchmark_obstacle_avoid_problem.simulation_config
   examples.benchmark_obstacle_avoid_problem.config
   examples.benchmark_obstacle_avoid_problem.main
   examples.benchmark_obstacle_avoid_problem.plot_config
   examples.benchmark_obstacle_avoid_problem.plot_results



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.benchmark_obstacle_avoid_problem.ex


.. py:function:: system_config()


.. py:function:: sample_config()


.. py:function:: simulation_config()


.. py:data:: ex
   

   

.. py:function:: config(sample)

   Experiment configuration variables.

   SOCKS uses sacred to run experiments in order to ensure repeatability. Configuration
   variables are parameters that are passed to the experiment, such as the random seed,
   and can be specified at the command-line.

   .. rubric:: Example

   To run the experiment normally, use:

       $ python -m <experiment>

   The full configuration can be viewed using:

       $ python -m <experiment> print_config

   To specify configuration variables, use `with variable=value`, e.g.

       $ python -m <experiment> with seed=123 system.time_horizon=5

   .. _sacred:
       https://sacred.readthedocs.io/en/stable/index.html



.. py:function:: main(seed, sigma, regularization_param, time_horizon, dynamic_programming, batch_size, heuristic, verbose, results_filename, no_plot, _log)

   Main experiment.


.. py:function:: plot_config(config, command_name, logger)


.. py:function:: plot_results(system, time_horizon, plot_cfg)

   Plot the results of the experiement.


