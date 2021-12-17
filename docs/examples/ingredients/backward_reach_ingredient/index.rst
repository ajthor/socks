:py:mod:`examples.ingredients.backward_reach_ingredient`
========================================================

.. py:module:: examples.ingredients.backward_reach_ingredient


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.backward_reach_ingredient.generate_tube
   examples.ingredients.backward_reach_ingredient.compute_test_point_ranges
   examples.ingredients.backward_reach_ingredient.generate_test_points



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.ingredients.backward_reach_ingredient.backward_reach_ingredient


.. py:data:: backward_reach_ingredient
   

   

.. py:function:: generate_tube(time_horizon, shape, bounds)

   Generate a stochastic reachability tube using config.

   This function computes a stochastic reachability tube using the tube configuration.

   :param env: The dynamical system model.
   :param bounds: The bounds of the tube. Specified as a dictionary.

   :returns: A list of spaces indexed by time.


.. py:function:: compute_test_point_ranges(env, test_points)


.. py:function:: generate_test_points(env, _log)

   Generate test points to evaluate the safety probabilities.


