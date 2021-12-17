:py:mod:`examples.example_separating_kernel`
============================================

.. py:module:: examples.example_separating_kernel

.. autoapi-nested-parse::

   Forward reachability example.

   This file demonstrates the forward reachability classifier on a set of dummy data. Note
   that the data is not taken from a dynamical system, but can easily be adapted to data
   taken from system observations via a simple substitution. The reason for the dummy data
   is to showcase the technique on a non-convex forward reachable set.

   .. rubric:: Example

   To run the example, use the following command:

       $ python -m examples.forward_reach.forward_reach

   .. [1] `Learning Approximate Forward Reachable Sets Using Separating Kernels, 2021
           Adam J. Thorpe, Kendric R. Ortiz, Meeko M. K. Oishi
           Learning for Dynamics and Control,
           <https://arxiv.org/abs/2011.09678>`_



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.example_separating_kernel.forward_reach_config
   examples.example_separating_kernel.config
   examples.example_separating_kernel.main
   examples.example_separating_kernel.plot_config
   examples.example_separating_kernel.plot_results



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.example_separating_kernel.ex


.. py:function:: forward_reach_config()


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



.. py:function:: main(seed, sigma, regularization_param, sample_size, results_filename, no_plot, _log)

   Main experiment.


.. py:function:: plot_config(config, command_name, logger)


.. py:function:: plot_results(results_filename, plot_cfg)

   Plot the results of the experiement.


