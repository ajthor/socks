***********************************************
:py:mod:`~gym_socks.algorithms.reach.kernel_sr`
***********************************************

Stochastic reachability algorithm benchmarks.

Parameters
==========

The benchmark parameters have the following default values. When they are not varied as
part of the benchmark, these values are held constant.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Experiment Parameter
      - Default Value
    * - ``dimensionality``
      - 2
    * - ``sample_size``
      - 1000
    * - ``test_sample_size``
      - 1000
    * - ``time_horizon``
      - 10

Results
=======

In the following figures, the blue line indicates the mean, the black dots indicate the
actual data, and the error bars indicate the 95% confidence interval.

.. figure:: stoch_reach_dimensionality_vs_time.png
    :scale: 100 %

    System dimensionality vs. computation time.

.. figure:: stoch_reach_sample_size_vs_time.png
    :scale: 100 %

    Sample size vs. computation time.

.. figure:: stoch_reach_test_sample_size_vs_time.png
    :scale: 100 %

    Test sample size vs. computation time.

.. figure:: stoch_reach_time_horizon_vs_time.png
    :scale: 100 %

    Time horizon vs. computation time.