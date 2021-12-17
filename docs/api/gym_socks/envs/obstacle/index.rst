:py:mod:`~gym_socks.envs.obstacle`
==================================

.. py:module:: gym_socks.envs.obstacle


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.obstacle.BaseObstacle




.. py:class:: BaseObstacle

   Bases: :py:obj:`gym_socks.envs.core.BaseDynamicalObject`, :py:obj:`abc.ABC`

   Base obstacle class.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from BaseObstacle.

   All obstacles inherit from BaseDynamicalObject, meaning they must implement the
   `step`, `reset`, `render`, `close`, and `seed` methods in their custom
   implementations. This is necessary so that the `World` class in
   `gym_socks.envs.world` can register them as objects in the world environment.

   In addition, all obstacles must implement an `intersects` method, which returns true
   if the state of the dynamical system intersects with the obstacle. For instance,
   this could identify a collision, an unsafe region, or a non-violable constraint.


   .. py:method:: intersects(self, time, state)
      :abstractmethod:

      Check if the state intersects the obstacle.

      This function is used to check whether or not the state of a dynamical system intersects the given obstacle. It should return True if the system intersects the obstacle and False otherwise.

      :param time: The time step of the simulation.
      :param state: The state of the dynamical system.

      :returns: A boolean indicator.
