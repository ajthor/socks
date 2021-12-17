:py:mod:`~gym_socks.algorithms.control.control_common`
======================================================

.. py:module:: gym_socks.algorithms.control.control_common

.. automodule:: gym_socks.algorithms.control.control_common

Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.control.control_common.compute_solution



.. py:function:: compute_solution(C, D = None, heuristic = False)

   Compute the solution to the LP.

   Computes a solution to the linear program, choosing either to delegate to the unconstrained or constrained solver depending on whether D is `None`.

   :param C: Array holding values of the cost function evaluated at sample points.
   :param D: Array holding values of the constraint function evaluated at sample points.
   :param heuristic: Whether to compute the heuristic solution.

   :returns: Probability vector.
   :rtype: gamma
