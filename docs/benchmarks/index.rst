**********
Benchmarks
**********

Several benchmarks are provided for the algorithms in SOCKS using `sacred
<https://github.com/IDSIA/sacred>`_ as an experimental framework. Sacred enables
repeatability by specifying a configuration, controlling randomness, and tracking
experiment runs.

.. toctree::
    :hidden:
    :maxdepth: 1

    kernel_sr/index
    kernel_sr_max/index


:doc:`kernel_sr/index`
    Stochastic reachability benchmarks.

:doc:`kernel_sr_max/index`
    Maximal stochastic reachability benchmarks.


Running Benchmarks
==================

.. tip::

    It is strongly recommended to read the `sacred documentation
    <https://sacred.readthedocs.io/en/stable/index.html>`_ before using the benchmarks.

.. note::

    We are still investigating the best way to implement benchmarks.

    See :doc:`/contributing/new_benchmarks` for more information.


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

We have implemented a custom runner for benchmarks.

The runner calls the algorithm repeatedly until the result returned by the benchmark is
within the 95% confidence interval for the data. This ensures the results are consistent
and statistically accurate.

The code runs once, without recording the result, in order to initiate the caching and
to warm up the CPU. The first run of a python script is known to be slow, since the code
and resources have not yet been loaded into memory by the CPU. Thus, the first, initial
run of the algorithm helps ensure accurate results. For further assurance, we run the
benchmark suite twice before posting the results.

Important Points
----------------

* The sample generation is **not** included as part of the measured time. Since the
  algorithms are data-driven, having prior data is generally considered the normal mode
  of operation. Thus, we do not report the simulated data generation process time in the
  results.
* The number of times the algorithms are run varies depending on how long it takes for
  the results to become statistically accurate.
* We have implemented several caching mechanisms in the code to speed up benchmark runs.
  These are used outside the measured code, so they should have no effect on the results
  presented.
* The error bars in the plots display the 95% confidence interval for the data, but are
  generally conservative (meaning wider than is probably true). In addition, it is
  important to understand that the empirical mean displayed by the data may lie outside
  the confidence interval, which is to be expected in certain cases.