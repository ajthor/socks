**************************
Neural Network vs. Kernels
**************************

Several recent papers have begun to dive into the implicit connections between neural
networks and kernel methods. They claim that understanding kernel methods will be
critical in order to understand and overcome some of the theoretical roadblocks that
have cropped up at the forefront of neural network research.

.. note::

    The connection between these two competing techniques has been studied extensively, so here we describe only the basic case of a single-layer neural network.

Feature Maps
============

You may have heard the term "feature" in learning literature, as in, machine learning
attempts to learn data *features* (which can potentially be described as learning the
underlying patterns in the data).

In practice, this often amounts to learning whether the data fits a set of pre-defined
functions :math:`\phi_{i}`, :math:`i = 1, 2, \ldots`, and we combine these features
using some sort of linear combination,

.. math::

    f(x) = \sum_{i=1}^{M} \alpha_{i} \phi_{i}(x)

In kernel methods, we have a very simple way of defining features via the kernel. In
effect, the features are simply the kernel function fixed at a number of pre-specified
points, i.e. :math:`\phi_{i}(\cdot) = k(x_{i}, \cdot)`.

.. math::

    f(x) = \sum_{i=1}^{M} \alpha_{i} k(x_{i}, x)

There are a number of well-known kernels that have been found, and are typically used in
kernel methods, such as the constant kernel :math:`k(x, x') = a`, linear kernel
:math:`k(x, x') = \langle x, x' \rangle`, polynomial kernel :math:`k(x, x') = (\langle
x, x' \rangle + c)^{d}`, and the Gaussian (RBF) kernel :math:`k(x, x') = \exp(-\lVert x
- x' \rVert^{2}/2 \sigma^{2})`.

We can also combine these kernels as needed, for example by adding them together,
scaling them, or multiplying two kernels, in order to create new valid kernels. This
allows us to define the "features" easily via the kernel.

.. hint::

    What this also means is, a kernel function can itself be a linear combination of
    kernels.

Approximation via Features
--------------------------

We can describe the learning problem in terms of features, meaning for a set of
pre-defined features :math:`\lbrace \phi_{i} \rbrace_{i=1}^{M}`, we seek to learn the
coefficients :math:`\alpha_{i}` of a linear combination of feature maps,

.. math::

    f(x) = \alpha_{1} \phi_{1}(x) + \cdots + \alpha_{M} \phi_{M}(x)

This is exactly the same as the kernel learning problem that we have described before.
In short, whether we call the kernels "features" or vice versa, we are essentially using
the same techniques that we have already described in the first part of the
:doc:`tutorial </guide/kernel_tutorial/tutorial>`.

However, it is important to note that **not all feature maps are valid kernels**. Some
features are complex, especially in the case of deep neural networks or convolutional
networks, which means not all neural networks are directly relatable to kernel methods.
Nevertheless, in certain cases, these two techniques are virtually identical.

Single-Layer Neural Networks
============================

In a single-layer neural network, we take an input vector :math:`x`, and compute a
linear transformation of the data, :math:`W x + b`, where :math:`W` is usually described
as a matrix of "weights", and :math:`b` is called a "bias" vector.

We then pass the result through a nonlinear transformation :math:`g`, which is called
the "activation" function. At the final step, we may then compute some linear
combination of the "activations" in order to compute the predicted value of the function
approximation.

.. math::

    f(x) = \sum_{i=1}^{M} \beta_{i} g_{i}(W x + b)

We specify the activation functions explicitly, and the neural network learning problem
is defined as learning the weights :math:`W`, the bias :math:`b`, and the linear
combination coefficients :math:`\beta_{i}`. Thanks to a technique called
*backpropagation*, we can directly compute the gradient of these values with respect to
the error. This means we can use an iterative training scheme called gradient descent,
where the gradient is used to update the weights, biases, and coefficients of the neural
network in order to iteratively improve the neural network approximation.

However, it is important to note that this procedure does not guarantee that we will
converge to a perfect solution. In practice, if we do not take sufficiently small steps
when we update the neural network parameters, we may fail to minimize the error of the
neural network approximator.

In order to describe a single-layer neural network in terms of features, we define the
feature map to be the activations of the neural network.

.. math::

    \varphi_{i}(x) = g_{i}(W x + b)

Then, it is easy to see that the neural network follows the same scheme as the kernel
learning problem when viewed from the last layer.

.. math::

    f(x) = \beta_{1} \varphi_{1}(x) + \cdots + \beta_{M} \varphi_{M}(x)

This is an important insight, since it gives us a way to relate the kernel features
:math:`\phi_{i}` and the neural network features :math:`\varphi_{i}`. In certain cases,
these feature maps may be directly related, meaning what we are doing in neural network
learning problems is effectively "learning the kernel".

.. note::

    Note that deep neural networks and convolutional networks are different, and the
    feature maps for these networks are significantly more complex. These networks
    typically do not have a simple analog to kernel methods.

Practical Differences
=====================

There are also some practical differences between neural networks and kernel methods
that we outline below.

Input Dimensionality
--------------------

Note that the dimensionality of the input vector plays a role in the complexity of a
neural network. If the dimensionality of the input vector :math:`x` is large, the weight
matrix :math:`W` will also have very high dimensionality. This can make maintaining and
updating the weights challenging, since we need to explicitly store and compute the
gradient updates for a high-dimensional matrix.

Kernel methods do not suffer from this problem, since the data is projected into a
high-dimensional function space, and we can use the :doc:`kernel trick
</guide/kernel_tutorial/rkhs_tutorial>`. This means we only need
to evaluate the *kernel* over high-dimensional data, which is typically much easier
since it boils down to an inner product of vectors.

Sequential Data
---------------

Neural networks are particularly adept at handling data which is provided
*sequentially*, meaning we receive observations given one at a time from the true
function. Since we can update the parameters of the network as data becomes available,
we do not need to store all of the data that has been passed previously.

The solution for kernel methods, however, requires that we store a matrix that scales
with the amount of data. This is a major hurdle for kernel methods, since practically it
requires us to first collect a large sample, which we can then use to easily compute the
"best fit" solution.

In short, what this means is that we are presented with a tradeoff. Neural networks take
a long time to train, but are relatively cheap to store in memory once the parameters
have been learned. Kernel methods, on the other hand, face large memory storage
requirements, but are trained in a single calculation. Despite this, neural networks are
typically better calibrated for sequential data scenarios when a large amount of data is
involved.

.. seealso::

    Note that several techniques for reducing the computational complexity of kernel
    methods have been developed, such as Random Fourier Features, the Nystrom Method,
    and Gaussian matrix approximations. Check out the :doc:`/examples/index` page for
    a demonstrations of how these techniques work.

Theoretical Guarantees
----------------------

One major disadvantage of neural networks is the lack of theoretical guarantees related
to convergence of the solution and what is called "explainability", i.e. why the network
has a particular output in response to a given input. This is a deep topic, so we will
only cover it here briefly.

Basically, the training procedure for neural networks is not conducive to
"introspection". Tracking an output backwards through the network leads to a complex web
of interconnected weights and activations, so determining the reasons why a network may
output a poor prediction is an area of open research.

In addition, the training for neural networks is typically iterative, and the solution
can jump around or get caught in a local solution without ever converging to the true,
global solution. Mathematically, this is because the neural network learning problem is
*non-convex*.

Kernel learning, on the other hand, is a *convex* problem, meaning kernel-based
solutions do not suffer from local minima, and are guaranteed to find the "best fit" for
a given data set. In addition, kernel methods are typically "shallow", meaning they
offer better explainability in their outputs. In other words, we can always find which
features and training data contributed to a certain prediction.
