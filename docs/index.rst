SOCKS
=====

Release: |release|

|binder_link|_

SOCKS is a suite of algorithms for stochastic optimal control using kernel methods.

It runs on top of `OpenAI Gym <https://gym.openai.com>`_, and comes with several classic
controls Gym environments. In addition, it can integrate with many pre-existing Gym
environments.

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Contents

    GitHub Repo <https://github.com/ajthor/socks>
    installation

    /examples/index

    /api/index

Getting Started
---------------

Install SOCKS using ``pip``:

.. code-block:: shell

    pip install gym-socks

Check out the :doc:`installation` page for more detailed instructions.

Examples
~~~~~~~~

After installing, the best way to familiarize yourself with SOCKS is by checking out
several of the key examples.

.. grid:: 2
    :gutter: 1 1 2 3

    .. grid-item-card::
        :link: examples/reach/stoch_reach_maximal
        :link-type: doc

        **Maximal Stochastic Reachability**
        ^^^

        Compute a policy that maximizes the probability of remaining within a safe set
        and reaching a target set.

        +++
        :bdg-primary-line:`control`
        :bdg-primary-line:`reachability`

    .. grid-item-card::
        :link: examples/reach/forward_reach
        :link-type: doc

        **Forward Reachability**
        ^^^

        Compute a forward reachable set classifier.

        +++
        :bdg-primary-line:`reachability`

    .. grid-item-card::
        :link: examples/control/tracking
        :link-type: doc

        **Target Tracking Problem**
        ^^^

        Unconstrained stochastic optimal control.

        +++
        :bdg-primary-line:`control`

    .. grid-item-card::
        :link: examples/control/obstacle_avoid
        :link-type: doc

        **Obstacle Avoid**
        ^^^

        Constrained stochastic optimal control.

        +++
        :bdg-primary-line:`control`


SOCKS can be run using `binder <https://mybinder.org>`_, and all examples can be run
interactively using the |binder_link|_ link included at the top of the example pages.
If you downloaded the code from the `GitHub repo <https://github.com/ajthor/socks>`_,
you can also run the examples locally or using `docker <https://www.docker.com>`_ and
the included Dockerfile.

.. |binder_link| image:: https://mybinder.org/badge_logo.svg
.. _binder_link: https://mybinder.org/v2/gh/ajthor/socks/HEAD

User Guide
----------

Check out the user guide to use SOCKS in your own projects.

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    using_socks


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


In order to cite the toolbox, use the following bibtex entry:

.. code-block:: bibtex

    @inproceedings{thorpe2022hscc,
      title     = {{SOCKS}: A Kernel-Based Stochastic Optimal Control and Reachability Toolbox},
      authors   = {Thorpe, Adam J. and Oishi, Meeko M. K.},
      year      = {2022},
      booktitle = {Proceedings of the 25th ACM International Conference on Hybrid Systems: Computation and Control (submitted)}
    }
