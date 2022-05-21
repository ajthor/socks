*************
RKHS Tutorial
*************

In this tutorial, we dive a little bit deeper into the theory of a reproducing kernel
Hilbert space and give a brief overview of one of the main properties of an RKHS, known
as the *reproducing property*.

*So what is an RKHS, really?*
=============================

In the first part of the :doc:`tutorial </guide/kernel_tutorial/tutorial>`, we described
the main idea of a kernel and kernel methods. In this part of the tutorial, we are going
to give a brief overview of an RKHS, and explain what that means in basic terms.

.. admonition:: Did you know?

    It might surprise you to know that you have likely been working with Hilbert spaces
    all along without even knowing it. The most common Hilbert space is
    :math:`\mathbb{R}^{n}`, which means that if you've taken any high school math course
    or linear algebra, you've probably already had a basic introduction to Hilbert
    spaces.

To recap, we started with a kernel function :math:`k : \mathcal{X} \times \mathcal{X}
\to \mathbb{R}`, and we said that by fixing one of the parameters, we get a kernel that
is "centered" at a particular value. In other words, for some fixed :math:`x`, we get a
function, :math:`k(x, \cdot)`. If we do this for all possible values of :math:`x`, what
we get is a set of functions :math:`\lbrace k(x, \cdot) \mid x \in \mathcal{X} \rbrace`.
The set of all of these functions is what we are going to use to construct our RKHS.
However, depending on the space :math:`\mathcal{X}`, it is easy to see that there might
be infinitely many functions in the set (for example, if :math:`\mathcal{X} =
\mathbb{R}`).

Before we continue, we are going to review the basic mathematical idea of vector spaces.
A vector space is a set of mathematical objects that can be added together and
multiplied by a scalar value. The elements of a vector space are called *vectors*. The
important point here is that *any space that satisfies the properties of a vector space
can be called a vector space*, and the elements of this space are called vectors.

Looking ahead, what we would like to do is combine these functions and multiply them by
some scalar value in order to compute a function approximation,

.. math::
    :label: linear_combination

    f(x) = \sum_{i=1}^{\infty} \alpha_{i} k(x_{i}, x)

This means if we can add our kernel functions :math:`k(x, \cdot)` and scale them by some
scalar value :math:`\alpha`, then the functions :math:`k(x, \cdot)` are vectors.

In essence, this is exactly what we have in :eq:`linear_combination`.

What this means is that the RKHS is a vector space, but instead of an array of numbers,
we have *functions*. Nevertheless, all of the properties of vector spaces still apply.

Inner Products
--------------

Now, we can add two vectors (kernel functions) together and multiply them by scalars,
but we cannot "multiply" two vectors together yet. To do this, we define what is known
as the "inner" product. When we're dealing with the familiar concept of vectors in
:math:`\mathbb{R}^{n}`, we have a special operation we can perform called the "dot"
product, :math:`x^{\top} y`, :math:`x, y \in \mathbb{R}^{n}`.

Mathematically, this is simply a different way of writing the inner product between
vectors, which is typically denoted as:

.. math::

    \langle x, y \rangle, \quad x, y \in \mathbb{R}^{n}

In order to turn our set of functions :math:`\lbrace k(x, \cdot) \mid x \in \mathcal{X}
\rbrace` into a Hilbert space, we need to define an inner product. Basically, we need to
explicitly define what it means to "multiply" two vectors together. Luckily, this is
already defined for us by the kernel, which is what makes kernels so special.

When we take one element from our set :math:`k(x, \cdot)`, and we take the inner product
with another element from our set :math:`k(x', \cdot)`, what we get is,

.. math::
    :label: kernel_trick

    \langle k(x, \cdot), k(x', \cdot) \rangle = k(x, x')

This has another name in kernel methods, and is called the *"kernel trick"*.

.. note::

    Once we have defined an inner product, there are a few more mathematical steps to
    define an RKHS. For the interested reader, we need to ensure that the space is also
    complete, meaning the space contains all of its limit points. Mathematically, this
    means an RKHS :math:`\mathscr{H}` is defined as the closure of the span of kernel
    functions,

    .. math::

        \mathscr{H} = \overline{\mathrm{span} \lbrace k(x, \cdot) \mid x \in
        \mathcal{X} \rbrace}

Functions in an RKHS
--------------------

To recap, what this means is that we start with a kernel function, and we use that
kernel function to build a set :math:`\lbrace k(x, \cdot) \mid x \in \mathcal{X}
\rbrace`. The kernel function also gives us an *inner product* via :eq:`kernel_trick`.
So now we can add the functions in our set, scale them, and take the inner product.

Then, the functions in an RKHS are relatively straightforward to describe. In effect,
they are some linear combination of kernel functions, just like we saw in the previous
:doc:`tutorial </guide/kernel_tutorial/tutorial>`. What this means is that any function
in the RKHS looks like a sum of weighted kernel functions,

.. math::
    :label: rkhs_function

    f(\cdot) = \sum_{i=1}^{\infty} \alpha_{i} k(x_{i}, \cdot)

where the coefficients :math:`\alpha_{i}` are what differentiate them from each other.

The Reproducing Property
------------------------

This leads us to the final property of an RKHS that is perhaps the most critical. This
property is known as the **"reproducing property"**. In a nutshell, the reproducing
property tells us that we can use the inner product to evaluate a function in the RKHS
at a particular point. Mathematically, it looks like this:

.. math::

    f(x) = \langle f, k(x, \cdot) \rangle

In order to gain a better understanding of how this works, let's substitute in the
definition of a function in the RKHS from :eq:`rkhs_function` and use the inner product
in :eq:`kernel_trick`.

.. math::

    f(x) = \langle f, k(x, \cdot) \rangle
    = \biggl\langle \sum_{i=1}^{\infty} \alpha_{i} k(x_{i}, \cdot), k(x, \cdot)
    \biggr\rangle = \sum_{i=1}^{\infty} \alpha_{i} \langle k(x_{i}, \cdot), k(x, \cdot)
    \rangle = \sum_{i=1}^{\infty} \alpha_{i} k(x_{i}, x)

Now, you may be wondering how we can take the infinite sum of kernel functions in order
to compute the function. The short answer is that generally, we can't. However, not all
kernels lead to infinite sums, and the kernel trick is a useful tool here. In general,
though, this is why we have function *approximations*, where instead of taking an
infinite sum, we approximate the function as a **finite** sum of kernel functions at a
set of known points (our data).

Recap
=====

At the end of this tutorial, you should have some basic understanding of kernels and
reproducing kernel Hilbert spaces. You should be able to describe what the functions
inside the RKHS look like, and how the reproducing property allows us to evaluate
functions.

We've merely scratched the surface on reproducing kernel Hilbert spaces, though. While
the basic theory is fairly straightforward, there is a large amount of research in this
area that comes directly from learning theory, probability, and functional analysis.