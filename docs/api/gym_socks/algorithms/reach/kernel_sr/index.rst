:py:mod:`~gym_socks.algorithms.reach.kernel_sr`
===============================================

.. py:module:: gym_socks.algorithms.reach.kernel_sr

.. automodule:: gym_socks.algorithms.reach.kernel_sr

Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.reach.kernel_sr.KernelSR



Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.reach.kernel_sr.kernel_sr



.. py:function:: kernel_sr(S, T, time_horizon = None, constraint_tube = None, target_tube = None, problem = 'THT', regularization_param = None, kernel_fn=None, batch_size = None, verbose = False)

   Stochastic reachability using kernel distribution embeddings.

   Computes an approximation of the safety probabilities of the stochastic reachability
   problem using kernel methods.

   :param S: Sample of (x, u, y) tuples taken iid from the system evolution. The sample
             should be in the form of a list of tuples.
   :param T: Evaluation points to evaluate the safety probabilities at. Should be in the
             form of a 2D-array, where each row indicates a point.
   :param time_horizon: Number of time steps to compute the approximation.
   :param constraint_tube: List of spaces or constraint functions. Must be the same
                           length as `time_horizon`.
   :param target_tube: List of spaces or target functions. Must be the same length as
                       `time_horizon`.
   :param problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
                   problem and `"FHT"` specifies the first-hitting time problem.
   :param kernel_fn: Kernel function used by the approximation.
   :param regularization_param: Regularization parameter used in the solution to the
                                regularized least-squares problem.
   :param batch_size: The batch size for more memory-efficient computations. Omit this
                      parameter or set to `None` to compute without batch processing.
   :param verbose: Boolean flag to indicate verbose output.

   :returns: An array of safety probabilities of shape {len(T), time_horizon}, where each row
             indicates the safety probabilities of the evaluation points at a different time
             step.


.. py:class:: KernelSR(time_horizon = None, constraint_tube = None, target_tube = None, problem = 'THT', kernel_fn=None, regularization_param = None, batch_size = None, verbose = False, *args, **kwargs)

   Bases: :py:obj:`gym_socks.algorithms.algorithm.AlgorithmInterface`

   Stochastic reachability using kernel distribution embeddings.

   Computes an approximation of the safety probabilities of the stochastic reachability
   problem using kernel methods.

   :param time_horizon: Number of time steps to compute the approximation.
   :param constraint_tube: List of spaces or constraint functions. Must be the same
                           length as `time_horizon`.
   :param target_tube: List of spaces or target functions. Must be the same length as
                       `time_horizon`.
   :param problem: One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
                   problem and `"FHT"` specifies the first-hitting time problem.
   :param kernel_fn: Kernel function used by the approximation.
   :param regularization_param: Regularization parameter used in the solution to the
                                regularized least-squares problem.
   :param batch_size: The batch size for more memory-efficient computations. Omit this
                      parameter or set to `None` to compute without batch processing.
   :param verbose: Boolean flag to indicate verbose output.

   .. py:method:: fit(self, S)

      Run the algorithm.

      :param S: Sample of (x, u, y) tuples taken iid from the system evolution. The
                sample should be in the form of a list of tuples.

      :returns: Instance of KernelSR class.
      :rtype: self


   .. py:method:: predict(self, T)

      Predict.

      :param T: Evaluation points to evaluate the safety probabilities at. Should be in
                the form of a 2D-array, where each row indicates a point.

      :returns: An array of safety probabilities of shape {len(T), time_horizon}, where each
                row indicates the safety probabilities of the evaluation points at a
                different time step.
