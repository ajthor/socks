*****************
Adding Benchmarks
*****************

Currently, the examples featured on SOCKS are designed to showcase the features and
capabilities of the algorithms provided in the toolbox. However, we are also very
interested in developing benchmarks that offer comparisons with other existing,
published techniques and that show the performance of the algorithms (e.g. scalability or accuracy).

.. note::

    Benchmarks will be soon be reported in the documentation. However, we are still
    investigating the best way to implement benchmarks.

    Have an idea? Please reach out. We'd love to hear from you!

Current Plan
============

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
==============

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

The Runner
----------

Using python's coroutines (``yield from``) is an easy way to accumulate results from
experiments. We implement a generator that accepts results from the user and stores them
in a list, like so:

.. code-block:: python

    def _result_accumulator(config):
        results = []
        while True:
            result = yield config
            if result is None:
                return results
            results.append(result)


    def experiment_matrix(config, results):
        experiment_matrix = zip(*config.items())
        keys, values = experiment_matrix

        for v in values:
            try:
                _ = iter(v)
            except TypeError:
                raise TypeError("Config values must be iterables.")

        for experiment_config in product(*values):
            a = _result_accumulator(dict(zip(keys, experiment_config)))
            result = yield from a

            results.append(result)

Then, when we are done inserting new values into the results, we simply send ``None`` or
just iterate as usual.

.. code-block:: python

    >>> results = []
    >>> M = experiment_matrix(config, results)
    >>> for params in M:
    ...     M.send(1)  # Experiment result inserted into `results`.
    ...     M.send(2)  # Experiment result inserted into `results`.
    >>> print(results)
    [[1, 2], [1, 2], [1, 2], ..., [1, 2], [1, 2], [1, 2]]

However, there are a few things which are strange with this implementation. First, we
pass the results variable to the ``experiment_matrix`` as a parameter, which is then
added to using the results from the ``_result_accumulator``. This is somewhat strange,
since there is really no need to update the results within the generator per-se. We
could easily do all of this manually within the for loop using a second ``while`` loop.

Ideally, we want a generator, since we do not want the user to enumerate over every
single configuration and loop manually. Second, it would be great if the user was
"blind", meaning they accept the parameters blindly, run an experiment using those
parameters, and then report the results, and do that as many times as they receive
parameters.

Here is an implementation of that using a generator class.

.. code-block:: python

    class ExperimentMatrix(Generator):
        def __init__(self, config):
            matrix = zip(*config.items())
            self._keys, values = matrix

            for v in values:
                try:
                    _ = iter(v)
                except TypeError:
                    raise TypeError("Config values must be iterables.")

            self._configs = product(*values)
            self._values = values

            self._index = None
            self.results = [[] for _ in range(len(self))]

        def __next__(self):
            try:
                config = next(self._configs)

            except StopIteration:
                self.throw(StopIteration)

            else:
                if self._index is None:
                    self._index = 0
                else:
                    self._index += 1

                self._current = dict(zip(self._keys, config))

            return self._current

        def send(self, value):
            if value is None:
                # Advance the iterator.
                return self.__next__()

            else:
                # Handle the result and return the same config.
                self.results[self._index].append(value)
                if len(self.results[self._index]) >= 3:
                    return self.__next__()

                return self._current

        def throw(self, type=None, value=None, traceback=None):
            raise StopIteration

        def __len__(self):
            return reduce(mul, map(len, self._values), 1)

We can then use this like so:

.. code-block:: python

    >>> M = ExperimentMatrix(config)
    >>> result = None
    >>> try:
    ...     while True:
    ...         params = M.send(result)
    ...         result = 1
    ... except StopIteration:
    ...     pass
    >>> print(M.results)
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], ..., [1, 1, 1], [1, 1, 1], [1, 1, 1]]


Note that the class keeps sending the same parameters multiple times, and accumulates
results until it is satisfied. This is beneficial, since it allows us to let the
generator decide when to terminate execution rather than the user.

What's more, we can still iterate over all of the parameter sets using a ``for`` loop:

.. code-block:: python

    >>> M = ExperimentMatrix(config)
    >>> for params in M:
    ...     print(params)

We can use the same ``while`` loop syntax for the coroutine above, but we have to place
the logic of when to stop passing results into the ``_result_accumulator``, which is
slightly more difficult to configure and access.

.. code-block:: python
    :emphasize-lines: 10,11,12

    def _result_accumulator(config):
        results = []
        while True:
            result = yield config
            if result is None:
                return results
            results.append(result)

            # Stopping logic.
            if len(results) >= 3:
                return results


Have a comparison?
==================

If you have a comparison that you would like to include, please get in touch via the
`discussion <https://github.com/ajthor/socks/discussions>`_ board. We'd love to help
include this in the SOCKS toolbox.