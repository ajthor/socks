:py:mod:`~gym_socks.algorithms.control.kernel_control_fwd`
==========================================================

.. py:module:: gym_socks.algorithms.control.kernel_control_fwd

.. automodule:: gym_socks.algorithms.control.kernel_control_fwd

Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.control.kernel_control_fwd.KernelControlFwd



Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.control.kernel_control_fwd.kernel_control_fwd



.. py:function:: kernel_control_fwd(S, A, cost_fn=None, constraint_fn=None, heuristic = False, regularization_param = None, kernel_fn=None, verbose = True)

   Stochastic optimal control policy forward in time.

   Computes the optimal control action at each time step in a greedy fashion. In other
   words, at each time step, the policy optimizes the cost function from the current
   state. It does not "look ahead" in time.

   :param S: Sample taken iid from the system evolution.
   :param A: Collection of admissible control actions.
   :param cost_fn: The cost function. Should return a real value.
   :param constraint_fn: The constraint function. Should return a real value.
   :param heuristic: Whether to use the heuristic solution instead of solving the LP.
   :param regularization_param: Regularization prameter for the regularized least-squares
                                problem used to construct the approximation.
   :param kernel_fn: The kernel function used by the algorithm.
   :param verbose: Whether the algorithm should print verbose output.

   :returns: The policy.


.. py:class:: KernelControlFwd(cost_fn=None, constraint_fn=None, heuristic = False, regularization_param = None, kernel_fn=None, verbose = True, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.policy.BasePolicy`

   Stochastic optimal control policy forward in time.

   Computes the optimal control action at each time step in a greedy fashion. In other
   words, at each time step, the policy optimizes the cost function from the current
   state. It does not "look ahead" in time.

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
