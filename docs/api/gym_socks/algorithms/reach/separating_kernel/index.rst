:py:mod:`~gym_socks.algorithms.reach.separating_kernel`
=======================================================

.. py:module:: gym_socks.algorithms.reach.separating_kernel

.. autoapi-nested-parse::

   Separating kernel classifier.

   Separating kernel classifier, useful for forward stochastic reachability analysis.

   .. rubric:: References

   .. [1] `Learning sets with separating kernels, 2014
           De Vito, Ernesto, Lorenzo Rosasco, and Alessandro Toigo.
           Applied and Computational Harmonic Analysis 37(2)`_

   .. [2] `Learning Approximate Forward Reachable Sets Using Separating Kernels, 2021
           Adam J. Thorpe, Kendric R. Ortiz, Meeko M. K. Oishi
           Learning for Dynamics and Control,
           <https://arxiv.org/abs/2011.09678>`_



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.reach.separating_kernel.SeparatingKernelClassifier




.. py:class:: SeparatingKernelClassifier(kernel_fn=None, regularization_param=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.algorithms.algorithm.AlgorithmInterface`

   Separating kernel classifier.

   A kernel-based support classifier for unknown distributions. Given a set of data taken iid from the distribution, the `SeparatingKernelClassifier` constructs a kernel-based classifier of the support of the distribution based on the theory of separating kernels.

   .. note::

      The sample used by the classifier is from the marginal distribution, not the
      joint or conditional. Thus, the data should be an array of points organized such
      that each point occupies a single row in a 2D-array.

   :param kernel_fn: The kernel function used by the classifier.
   :param regularization_param: The regularization parameter used in the regularized
                                least-squares problem. Determines the smoothness of the solution.

   .. rubric:: Example

   >>> from gym_socks.algorithms.reach import SeparatingKernelClassifier
   >>> from gym_socks.kernel.metrics import abel_kernel
   >>> from functools import partial
   >>> kernel_fn = partial(abel_kernel, sigma=0.1)
   >>> classifier = SeparatingKernelClassifier(kernel_fn)
   >>> classifier.fit(S)
   >>> classifications = classifier.predict(T)

   .. py:method:: fit(self, X)

      Fit separating kernel classifier.

      :param X: Data drawn from distribution.

      :returns: Instance of SeparatingKernelClassifier
      :rtype: self


   .. py:method:: predict(self, T)

      Predict using the separating kernel classifier.

      :param T: Evaluation points where the separating kernel classifier is evaluated.

      :returns: Boolean indicator of classifier.
