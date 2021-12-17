:py:mod:`examples.benchmark_stochastic_reachability`
====================================================

.. py:module:: examples.benchmark_stochastic_reachability

.. autoapi-nested-parse::

   Stochastic reachability.

   This example shows the stochastic reachability algorithm.

   By default, the system is a double integrator (2D stochastic chain of integrators).

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.benchmark_stochastic_reachability

   .. [1] `Model-Free Stochastic Reachability
           Using Kernel Distribution Embeddings, 2019
           Adam J. Thorpe, Meeko M. K. Oishi
           IEEE Control Systems Letters,
           <https://arxiv.org/abs/1908.00697>`_



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.benchmark_stochastic_reachability.system_config
   examples.benchmark_stochastic_reachability.sample_config
   examples.benchmark_stochastic_reachability.backward_reach_config
   examples.benchmark_stochastic_reachability.config
   examples.benchmark_stochastic_reachability.main
   examples.benchmark_stochastic_reachability.plot_results
   examples.benchmark_stochastic_reachability.plot_config_3d
   examples.benchmark_stochastic_reachability.plot_results_3d



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.benchmark_stochastic_reachability.ex


.. py:function:: system_config()


.. py:function:: sample_config()


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



.. py:function:: main(seed, _log, sigma, regularization_param, time_horizon, backward_reach, batch_size, verbose, results_filename, no_plot)

   Main experiment.


.. py:function:: plot_results(plot_cfg, results_filename)

   Plot the results of the experiement.


.. py:function:: plot_config_3d(config, command_name, logger)


.. py:function:: plot_results_3d(plot_cfg, results_filename)


