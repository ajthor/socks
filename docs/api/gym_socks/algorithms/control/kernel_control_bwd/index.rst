:py:mod:`~gym_socks.algorithms.control.kernel_control_bwd`
==========================================================

.. py:module:: gym_socks.algorithms.control.kernel_control_bwd

.. automodule:: gym_socks.algorithms.control.kernel_control_bwd

Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.control.kernel_control_bwd.KernelControlBwd



Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.control.kernel_control_bwd.kernel_control_bwd



.. py:function:: kernel_control_bwd(S, A, time_horizon = None, cost_fn=None, constraint_fn=None, heuristic = False, regularization_param = None, kernel_fn=None, batch_size = None, verbose = True)

   Stochastic optimal control policy backward in time.

   Computes the optimal policy using dynamic programming. The solution computes an approximation of the value functions starting at the terminal time and working backwards. Then, when the policy is evaluated, it moves forward in time, optimizing over the value functions and choosing the action which has the highest "value".

   :param S: Sample taken iid from the system evolution.
   :param A: Collection of admissible control actions.
   :param cost_fn: The cost function. Should return a real value.
   :param constraint_fn: The constraint function. Should return a real value.
   :param heuristic: Whether to use the heuristic solution instead of solving the LP.
   :param regularization_param: Regularization prameter for the regularized least-squares
                                problem used to construct the approximation.
   :param kernel_fn: The kernel function used by the algorithm.
   :param verbose: Whether the algorithm should print verbose output.


.. py:class:: KernelControlBwd(time_horizon = None, cost_fn=None, constraint_fn=None, heuristic = False, regularization_param = None, kernel_fn=None, batch_size = None, verbose = True, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.policy.BasePolicy`

   Stochastic optimal control policy backward in time.

   Computes the optimal policy using dynamic programming. The solution computes an approximation of the value functions starting at the terminal time and working backwards. Then, when the policy is evaluated, it moves forward in time, optimizing over the value functions and choosing the action which has the highest "value".

   :param S: Sample taken iid from the system evolution.
   :param A: Collection of admissible control actions.
   :param cost_fn: The cost function. Should return a real value.
   :param constraint_fn: The constraint function. Should return a real value.
   :param heuristic: Whether to use the heuristic solution instead of solving the LP.
   :param regularization_param: Regularization prameter for the regularized least-squares
                                problem used to construct the approximation.
   :param kernel_fn: The kernel function used by the algorithm.
   :param verbose: Whether the algorithm should print verbose output.

   .. py:method:: train(self, S, A)

      Train the algorithm.

      :param S: Sample taken iid from the system evolution.
      :param A: Collection of admissible control actions.

      :returns: An instance of the KernelControlFwd algorithm class.
      :rtype: self


   .. py:method:: __call__(self, time=0, state=None, *args, **kwargs)

      Evaluate the policy.

      :returns: An action returned by the policy.
      :rtype: action
