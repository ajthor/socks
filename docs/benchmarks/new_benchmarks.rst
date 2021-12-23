Creating New Benchmarks
=======================

Currently, the examples featured on SOCKS are designed to showcase the features and
capabilities of the algorithms provided in the toolbox. However, we are also very
interested in developing benchmarks that offer comparisons with other existing,
published techniques and that show the performance of the algorithms (e.g. scalability or accuracy).

.. note::

    Benchmarks will be soon be reported in the documentation. However, we are still
    investigating the best way to implement benchmarks.

    Have an idea? Please reach out. We'd love to hear from you!

Current Plan
------------

* Implement the benchmarks using `sacred <https://github.com/IDSIA/sacred>`_. This
  allows for greater control over repeatability and experiment parameterization. As
  opposed to a single script per technique or benchmark, which would potentially require
  a lot of copy-paste and boilerplate, we see the advantage of being able to run
  experiments with changes passed via the command-line or through extensive
  configuration files. This negates the need to re-write benchmarks and promotes
  encapsulation.
* Write a "runner" class or script that takes an experiment matrix (such as running an
  experiment with several parameter values) and saves the result of each experiment to a
  matrix. This should allow for "warm-starting", meaning it runs an experiment or noop
  script to warm up the CPU, and then goes through the experiment matrix. The results
  are then stored (perhaps using sacred's storage options), and a final plot is
  generated using the data at the end. The goal is to isolate the experiment from every
  other piece of the benchmarking process.
* Write a "benchmark" class that performs comparisons between different techniques
  using the same principles as above.

Main Questions
--------------

* **How do we implement other techniques?** These often have their own dependencies or
  are written in other languages such as Matlab, which will need to be tracked
  independently. One potential way of handing this would be to use something like
  docker, but having a separate Dockerfile for each algorithm we wish to compare against
  seems excessive. The nuclear option would be to re-implement techniques as baselines,
  but this would potentially be unfair to techniques that rely upon lower-level
  languages such as C or C++. One possible advantage is that python can interface with C
  and C++ rather easily via extensions, but these are difficult to write.
* **How do we structure the runner?** My first instinct is to use sacred, and to
  specify the "experiments" individually, and inject parameters via config updates.
  However, this in itself will require substantial boilerplate, since each experiment
  will have its own set of parameter names and values. The advantage to this approach is
  that the benchmarks are encapsulated well, and will require very little modification
  between benchmarks.
* **How do we report the results in the docs?** Since the benchmarks will take a long
  time to run, we don't want to use the same technique that we used for the experiments,
  which are converted to interactive binder scripts. This is mainly due to the fact that
  the scripts are executed when the docs are generated, and may impose significant
  overhead in the readthedocs build process. Additionally, the binder images are not
  designed for intense, prolonged computation, meaning this may not be the best platform
  to run benchmarks anyway.

  * The simple option would be to save the output images and then simply insert them
    into the docs, but this would effectively mean that the benchmarks are
    non-interactive and the results would depend on the machine we run them on.

  * Another option is to run the benchmarks on a service like `CodeOcean
    <https://codeocean.com>`_, which offers powerful, cloud-based runners with limited
    computation time each month. This would require extra setup, but would not be
    insurmountable.

Have a comparison?
------------------

If you have a comparison that you would like to include, please get in touch via the
`discussion <https://github.com/ajthor/socks/discussions>`_ board. We'd love to help
include this in the SOCKS toolbox.