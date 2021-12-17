:py:mod:`examples.example_linear_id`
====================================

.. py:module:: examples.example_linear_id

.. autoapi-nested-parse::

   Linear system identification example.

   This file demonstrates the linear system identification algorithm.

   By default, it uses the CWH4D system dynamics. Try setting the regularization parameter
   lower for higher accuracy. Note that this can introduce numerical instability if set too
   low.

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.example_linear_id



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.example_linear_id.system_config
   examples.example_linear_id.sample_config
   examples.example_linear_id.simulation_config
   examples.example_linear_id.config
   examples.example_linear_id.main
   examples.example_linear_id.plot_config
   examples.example_linear_id.plot_results



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.example_linear_id.ex


.. py:function:: system_config()


.. py:function:: sample_config()


.. py:function:: simulation_config()


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



.. py:function:: main(simulation, seed, regularization_param, time_horizon, results_filename, no_plot, _log)

   Main experiment.


.. py:function:: plot_config(config, command_name, logger)


.. py:function:: plot_results(system, plot_cfg)

   Plot the results of the experiement.


