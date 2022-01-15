**********
Benchmarks
**********

.. toctree::
    :hidden:
    :maxdepth: 1

    stoch_reach/index
    stoch_reach_maximal/index


:doc:`stoch_reach/index`
    Stochastic reachability benchmarks.

:doc:`stoch_reach_maximal/index`
    Maximal stochastic reachability benchmarks.


Running Benchmarks
==================

Several benchmarks are provided for the algorithms in SOCKS using `sacred
<https://github.com/IDSIA/sacred>`_ as an experimental framework. Sacred enables
repeatability by specifying a configuration, controlling randomness, and tracking
experiment runs. It is strongly recommended to read the `sacred documentation
<https://sacred.readthedocs.io/en/stable/index.html>`_ before using the benchmarks.

.. note::

    We are still investigating the best way to implement benchmarks.

    See :doc:`/contributing/new_benchmarks` for more information.

Quick Start
===========

The benchmarks and examples can be run using the python 'module' syntax, i.e.:

.. code-block:: shell

    python -m <benchmark>

This uses the default parameters and configuration for the experiments. To view the
configuration for a particular experiment, use the ``print_config`` command after the
experiment name, like so:

.. code-block:: shell

    python -m <benchmark> print_config

You should see a description of all of the configuration variables that are available.
Most algorithm parameters are controlled by the sacred config, and sacred allows for
configuration updates to be specified at the command-line, making it easy to change
parameters at run-time without having to modify the experiment file. To specify
parameters, use the ``with`` syntax, followed by the configuration updates. For
instance, to specify a particular seed, use the following command:

.. code-block:: shell

    python -m <benchmark> with seed=0

You can also view the updates before running the experiment using the ``print_config``
command alongside the ``with <updates>`` syntax.

.. code-block:: shell

    python -m <benchmark> print_config with seed=0


Understanding Benchmarks
========================

We have implemented a custom runner for benchmarks. It runs the algorithm repeatedly
until the result returned by the benchmark is within the 95% confidence interval for the
data. This ensures the results are consistent and statistically accurate.

In addition, we have implemented several caching mechanisms in the code to speed up
benchmark runs. These are used outside the measured code, so they should have no effect
on the results presented.

Lastly, the code runs once, without recording the result, in order to initiate the
caching and to warm up the CPU. The first run of a python script is known to be slow,
since the code and resources have not yet been loaded into memory by the CPU. The first,
initial run of the algorithm helps ensure accurate results. For further assurance, we
run the benchmark suite twice.

Important Points
^^^^^^^^^^^^^^^^

* The sample generation is **not** included as part of the measured time. Since the
  algorithms are data-driven, having prior data is generally considered the normal mode
  of operation. Thus, we do not report the simulated data generation process time in the
  results.
* The number of times the algorithms are run varies depending on how long it takes for
  the results to become statistically accurate. In the worst case, the number of runs is
  capped at 256.
* The error bars in the plots display the 95% confidence interval for the data, but are
  conservative (meaning wider than is probably true). In addition, it is important to
  understand that the empirical mean displayed by the data may lie outside the
  confidence interval, which is to be expected in certain cases.