:py:mod:`examples.ingredients.tracking_ingredient`
==================================================

.. py:module:: examples.ingredients.tracking_ingredient


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.tracking_ingredient.compute_target_trajectory
   examples.ingredients.tracking_ingredient.make_cost



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.ingredients.tracking_ingredient.tracking_ingredient


.. py:data:: tracking_ingredient
   

   

.. py:function:: compute_target_trajectory(time_horizon, path_amplitude, path_period)

   Computes the target trajectory to follow.

   The default trajectory is a V-shaped path based on a triangle function. The amplitude and period are set by the config.

   :param amplitude: The amplitude of the triangle function.
   :param period: The period of the triangle function.
   :param sampling_time: The sampling time of the dynamical system. Ensures that there
                         are a number of points in the target trajectory equal to the number of time
                         steps in the simulation.
   :param time_horizon: The time horizon of the dynamical system. Ensures that there are
                        a number of points in the target trajectory equal to the number of time
                        steps in the simulation.

   :returns: The target trajectory as a list of points.
   :rtype: target_trajectory


.. py:function:: make_cost(target_trajectory, norm_order, squared)


