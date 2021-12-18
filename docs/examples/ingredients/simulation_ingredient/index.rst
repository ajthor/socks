:py:mod:`~examples.ingredients.simulation_ingredient`
=====================================================

.. py:module:: examples.ingredients.simulation_ingredient

.. autoapi-nested-parse::

   Simulation ingredient.

   Used for experiments where the system needs to be simulated, such as after computing a
   policy via one of the included control algorithms.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.simulation_ingredient.config
   examples.ingredients.simulation_ingredient.simulate_system



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.ingredients.simulation_ingredient.simulation_ingredient


.. py:data:: simulation_ingredient




.. py:function:: config()

   Simulation configuration variables.


.. py:function:: simulate_system(time_horizon, env, policy, initial_condition, _log)

   Simulate the system from a given initial condition.
