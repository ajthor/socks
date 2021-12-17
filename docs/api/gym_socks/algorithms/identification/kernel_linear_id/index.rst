:py:mod:`~gym_socks.algorithms.identification.kernel_linear_id`
===============================================================

.. py:module:: gym_socks.algorithms.identification.kernel_linear_id

.. autoapi-nested-parse::

   Kernel-based linear system identification.

   The algorithm uses a concatenated state space representation to compute the state and
   input matrices given a sample of system observations. Uses the matrix inversion lemma
   and a linear kernel to compute the linear relationship between observations.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.identification.kernel_linear_id.KernelLinearId



Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.identification.kernel_linear_id.kernel_linear_id



.. py:function:: kernel_linear_id(S, regularization_param = None, verbose = False)

   Stochastic linear system identification using kernel distribution embeddings.

   Computes an approximation of the state and input matrices, as well as an estimate of
   the stochastic disturbance mean given a sample of system observations.

   :param S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
             should be in the form of a list of tuples.
   :param regularization_param: Regularization parameter used in the solution to the
                                regularized least-squares problem.
   :param verbose: Boolean flag to indicate verbose output.

   :returns: The fitted model computed using the linear identification algorithm.


.. py:class:: KernelLinearId(regularization_param = None, verbose = False, *args, **kwargs)

   Bases: :py:obj:`gym_socks.algorithms.algorithm.AlgorithmInterface`

   Stochastic linear system identification using kernel distribution embeddings.

   Computes an approximation of the state and input matrices, as well as an estimate of
   the stochastic disturbance mean given a sample of system observations.

   :param regularization_param: Regularization parameter used in the solution to the
                                regularized least-squares problem.
   :param verbose: Boolean flag to indicate verbose output.

   .. py:method:: state_matrix(self)
      :property:

      The estimated state matrix.


   .. py:method:: input_matrix(self)
      :property:

      The estimated input matrix.


   .. py:method:: disturbance_mean(self)
      :property:

      The estimated stochastic disturbance mean.


   .. py:method:: fit(self, S)

      Run the algorithm.

      :param S: Sample of (x, u, y) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.

      :returns: Instance of KernelLinearId class.
      :rtype: self


   .. py:method:: predict(self, T, U = None)

      Predict.

      :param T: State vectors. Should be in the form of a 2D-array, where each row
                indicates a state.
      :param U: Input vectors. Should be in the form of a 2D-array, where each row
                indicates an input.

      :returns: The predicted resulting states after propagating through the estimated
                system dynamics.
