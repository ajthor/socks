:py:mod:`~examples.example_monte_carlo`
=======================================

.. py:module:: examples.example_monte_carlo

.. autoapi-nested-parse::

   Stochastic reachability using Monte-Carlo.

   This example shows the Monte-Carlo stochastic reachability algorithm.

   By default, the system is a double integrator (2D stochastic chain of integrators).

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.example_monte_carlo



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.example_monte_carlo.system_config
   examples.example_monte_carlo.backward_reach_config
   examples.example_monte_carlo.config
   examples.example_monte_carlo.main
   examples.example_monte_carlo.plot_results
   examples.example_monte_carlo.plot_config_3d
   examples.example_monte_carlo.plot_results_3d



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.example_monte_carlo.ex


.. py:function:: system_config()


.. py:function:: backward_reach_config()


.. py:data:: ex




.. py:function:: config()

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



.. py:function:: main(seed, num_iterations, time_horizon, backward_reach, verbose, results_filename, no_plot, _log)

   Main experiment.


.. py:function:: plot_results(plot_cfg, results_filename)

   Plot the results of the experiement.


.. py:function:: plot_config_3d(config, command_name, logger)


.. py:function:: plot_results_3d(plot_cfg, results_filename)
