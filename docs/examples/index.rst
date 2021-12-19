Examples
========

.. py:module:: examples

Several examples and benchmarks are provided for the algorithms in SOCKS using `sacred
<https://github.com/IDSIA/sacred>`_ as an experimental framework. Sacred enables
repeatability by specifying a configuration, controlling randomness, and tracking
experiment runs. It is strongly recommended to read the `sacred documentation
<https://sacred.readthedocs.io/en/stable/index.html>`_ before using the examples.

Quick Start
-----------

The benchmarks and examples can be run using the python 'module' syntax, i.e.:

.. code-block:: shell

    python -m <examples.experiment_name>

This uses the default parameters and configuration for the experiments. To view the
configuration for a particular experiment, use the ``print_config`` command after the
experiment name, like so:

.. code-block:: shell

    python -m <examples.experiment_name> print_config

You should see a description of all of the configuration variables that are available.
Most algorithm parameters are controlled by the sacred config, and sacred allows for
configuration updates to be specified at the command-line, making it easy to change
parameters at run-time without having to modify the experiment file. To specify
parameters, use the ``with`` syntax, followed by the configuration updates. For
instance, to specify a particular seed, use the following command:

.. code-block:: shell

    python -m <examples.experiment_name> with seed=0

You can also view the updates before running the experiment using the ``print_config``
command alongside the ``with <updates>`` syntax.

.. code-block:: shell

    python -m <examples.experiment_name> print_config with seed=0

Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   ingredients/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   benchmark_cwh_problem/index.rst
   benchmark_maximal_stochastic_reachability/index.rst
   benchmark_obstacle_avoid_problem/index.rst
   benchmark_stochastic_reachability/index.rst
   benchmark_tracking_problem/index.rst
   example_linear_id/index.rst
   example_monte_carlo/index.rst
   example_separating_kernel/index.rst
