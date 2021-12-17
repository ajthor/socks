:py:mod:`examples.ingredients.system_ingredient`
================================================

.. py:module:: examples.ingredients.system_ingredient


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.system_ingredient.make_system
   examples.ingredients.system_ingredient.set_system_seed
   examples.ingredients.system_ingredient.print_info



Attributes
~~~~~~~~~~

.. autoapisummary::

   examples.ingredients.system_ingredient.system_ingredient


.. py:data:: system_ingredient
   

   

.. py:function:: make_system(system_id, sampling_time, _config, _log)

   Construct an instance of the system.

   :param system_id: System identifier string.
   :param time_horizon: The time horizon of the system.
   :param sampling_time: The sampling time of the system.

   :returns: An instance of the dynamical system model.


.. py:function:: set_system_seed(seed, env, _log)

   Set the random seed.

   :param seed: The random seed for the experiment.
   :param env: The dynamical system model.


.. py:function:: print_info(system_id)

   Print system info.

   Prints information of the system specified by `system_id` to the screen.

   :param system_id: System identifier string.


