:py:mod:`~examples.benchmark_cwh_problem`
=========================================

.. py:module:: examples.benchmark_cwh_problem

.. autoapi-nested-parse::

   Satellite rendezvous and docking.

   This example demonstrates the optimal controller synthesis algorithm on a satellite rendezvous and docking problem with CWH dynamics.

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.benchmark_cwh_problem

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

   examples.benchmark_cwh_problem.system_config
   examples.benchmark_cwh_problem.sample_config
   examples.benchmark_cwh_problem.simulation_config
   examples.benchmark_cwh_problem.config
   examples.benchmark_cwh_problem.main
   examples.benchmark_cwh_problem.plot_config
   examples.benchmark_cwh_problem.plot_results
   examples.benchmark_cwh_problem.plot_sample



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.benchmark_cwh_problem.ex


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



.. py:function:: main(seed, sigma, regularization_param, time_horizon, dynamic_programming, batch_size, heuristic, verbose, results_filename, no_plot, simulation, _log)

   Main experiment.


.. py:function:: plot_config(config, command_name, logger)


.. py:function:: plot_results(plot_cfg, _log)

   Plot the results of the experiement.


.. py:function:: plot_sample(seed, plot_cfg)

   Plot a sample taken from the system.
