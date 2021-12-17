:py:mod:`~gym_socks.envs.QUAD20`
================================

.. py:module:: gym_socks.envs.QUAD20

.. autoapi-nested-parse::

   Quadrotor system.

   .. rubric:: References

   .. [1] `ARCH-COMP20 Category Report:
           Continuous and Hybrid Systems with Nonlinear Dynamics
           <https://easychair.org/publications/open/nrdD>`_



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.QUAD20.QuadrotorEnv




.. py:class:: QuadrotorEnv(seed=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   Quadrotor system.

   The quadrotor system is a high-dimensional (12D) system. The states are the
   position, velocity, and angles of the system, and the inputs are the torques on the
   angles.


   .. py:method:: gravitational_acceleration(self)
      :property:


   .. py:method:: radius_center_mass(self)
      :property:


   .. py:method:: rotor_distance(self)
      :property:


   .. py:method:: rotor_mass(self)
      :property:


   .. py:method:: center_mass(self)
      :property:


   .. py:method:: generate_disturbance(self, time, state, action)

      Generate disturbance.


   .. py:method:: dynamics(self, time, state, action, disturbance)

      Dynamics for the system.
