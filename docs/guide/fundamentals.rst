************
Fundamentals
************

What is SOCKS?
==============

SOCKS is a data-driven stochastic optimal control toolbox that uses a statistical
learning technique known as kernel methods to solve constrained stochastic optimal
control problems.

The main features of SOCKS are:

* Algorithms to compute control policies, approximate dynamics, and analyze stochastic
  systems.
* Functions to model dynamical systems and generate samples.
* Implementations of kernel-based learning techniques.

.. admonition:: So what are kernel methods?

    Kernel methods are a class of statistical learning techniques that use the theory of
    high-dimensional function spaces called reproducing kernel Hilbert spaces (RKHS) to
    do regression, classification, and estimation. Unlike neural networks, which operate
    on a stream of data and learn iteratively using gradient descent, kernel methods use
    a sample of data collected "up front" and are typically non-iterative. This makes
    kernel methods fast, but less sample-efficient.

What is Stochastic Optimal Control?
===================================

A dynamical system is an object that changes through time. For example, a dynamical
system could be a vehicle such as a car, plane, or spacecraft, an electrical circuit or
power grid, or a biological system or chemical reaction. We use a set of mathematical
equations and variables to describe how the system evolves over time, and in control
theory, we typically have the ability to manipulate some of these variables in order to
affect the future evolution of the system.

In stochastic control, the system has an element of randomness (stochasticity) attached
to it. I.e., the evolution of the system is affected by random forces, which means we
cannot predict the exact evolution of the system. Instead, we are only able to predict
where the system will end up with a certain probability or likelihood.

Optimal control seeks to compute the inputs to a system that guide the system to a
desired state while minimizing (or maximizing) a pre-specified cost function. A simple
example could be reaching a goal in the shortest amount of time or by using the least
amount of fuel. In stochastic optimal control, we must also account for the stochastic
effects which are present in the system. For instance, we may need to account for wind
in order to safely land a plane.

To complicate matters further, we may need to account for certain operating constraints
that limit the control actions we can take. For example, we may want to safely guide a
car to a destination, meaning we want to remain on the road, avoid pedestrians and other
vehicles, and respect traffic laws. This is a very difficult problem in controls, since
the stochasticity of the system means we can only guarantee that we will be able to
satisfy constraints with a certain likelihood.

A stochastic optimal control problem can be written mathematically as,

.. math::

    \begin{align}
        \min_{\pi} \quad & \int_{\mathcal{U}} \int_{\mathcal{X}} f_{0}(y, u)
        Q(\mathrm{d} y \mid x, u) \pi(\mathrm{d} u \mid x) \\
        \textnormal{s.t.} \quad & \int_{\mathcal{U}} \int_{\mathcal{X}} f_{i}(y, u)
        Q(\mathrm{d} y \mid x, u) \pi(\mathrm{d} u \mid x) \leq 0,
        \quad i = 1, \ldots, n
    \end{align}

In short, we seek to minimize the expectation of the function :math:`f_{0}` subject to
the expectation constraints :math:`f_{1}, \ldots, f_{n}`. The function :math:`Q` is
called a stochastic kernel, and describes the probability of ending up in a future state
given that the system is currently in state :math:`x` and we apply a control action
:math:`u`. The control action :math:`u` is typically chosen by a function :math:`\pi`
called a policy that chooses an action based on the state of the system :math:`x`. In
other words, if we know what the state of the system is, we can strategically choose the
control actions that guide the system to a desired state with high likelihood.

.. note::

    A stochastic kernel is the mathematical way to define a conditional distribution. In
    reinforcement learning, this is commonly known as a transition kernel, or a state
    transition function.

To make matters worse, we typically do not have full knowledge about the nature
of the stochasticity, meaning we do not know all of the properties of the random
effects. This makes things significantly more challenging, since we cannot simply
"solve" the optimization problem. Instead, we have to learn the dynamics of the system
(:math:`Q`) in order to decide how to control it.

Data-Driven Control
===================

One way to approach this problem is using *data-driven control*, which uses observations
of the system evolution to learn an approximation of the system dynamics. If we observe
the system under different control actions and in different scenarios, we can learn how
it moves or changes and build an approximation of the dynamical equations and the
nature of the stochasticity. Then, we can learn how to control the approximate system.

Formally, consider a discrete-time system with dynamics

.. math::
    :label: dynamics

    x_{t+1} = f(x_{t}, u_{t}, w_{t}),

where :math:`x_{t}` is the state of the system at time :math:`t` and is an element of
the state space :math:`\mathcal{X}`, :math:`u_{t}` is the control action applied to the
system at time :math:`t` and is an element of the control space :math:`\mathcal{U}`, and
:math:`w_{t}` is a random variable that represents process noise. We can equivalently
represent the system dynamics in :eq:`dynamics` using a stochastic kernel :math:`Q`,
which assigns a conditional probability measure :math:`Q(\cdot \mid x, u)` to each
:math:`x` and :math:`u`.

A sample (otherwise known as a dataset) is a sequence of observations taken from a
stochastic kernel. Given system dynamics as in :eq:`dynamics`, a system observation
consists of the starting system state, an applied control action, and the resulting
state. A sample :math:`\mathcal{S}` of size :math:`n \in \mathbb{N}` is a collection of
observations.

.. math::

    \mathcal{S} = \lbrace (x_{i}, u_{i}, x_{i}{}^{\prime}) \rbrace_{i=1}^{n}

where :math:`x_{i}` and :math:`u_{i}` are sampled i.i.d. from distributions on the state
space :math:`\mathcal{X}` and control space :math:`\mathcal{U}`, and
:math:`x_{i}{}^{\prime}` is a state in :math:`\mathcal{X}` that denotes the
state at the next time step, computed using the dynamics :math:`f`.
