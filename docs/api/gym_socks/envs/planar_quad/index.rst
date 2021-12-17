:py:mod:`~gym_socks.envs.planar_quad`
=====================================

.. py:module:: gym_socks.envs.planar_quad

.. autoapi-nested-parse::

   Planar quadrotor system.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.planar_quad.PlanarQuadrotorEnv




.. py:class:: PlanarQuadrotorEnv(seed=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   Planar quadrotor system.

   A planar quadrotor is quadrotor restricted to two dimensions. Similar to the OpenAI gym lunar lander benchmark, the planar quadrotor is a bar with two independent rotors at either end. Inputs are the trust of the rotors, and apply a torque to the bar. The system is also subject to gravitational forces.


   .. py:method:: gravitational_acceleration(self)
      :property:


   .. py:method:: rotor_distance(self)
      :property:


   .. py:method:: total_mass(self)
      :property:


   .. py:method:: inertia(self)
      :property:


   .. py:method:: step(self, action, time=0)

      Step function defined by OpenAI Gym.

      Advances the system forward one time step.

      :param time: Time of the simulation. Used primarily for time-varying systems.
      :param action: Action (input) applied to the system at the current time step.

      :returns:

                The observation vector. Generally, it is the state of the system
                    corrupted by some measurement noise. If the system is fully observable,
                    this is the state of the system at the next time step.
                cost: The cost (reward) obtained by the system for taking action u in state
                    x and transitioning to state y. In general, this is not typically used with `DynamicalSystem` models.
                done: Flag to indicate the simulation has terminated. Usually toggled by
                    guard conditions, which terminates the simulation if the system
                    violates certain operating constraints.
                info: Extra information.
      :rtype: obs


   .. py:method:: generate_disturbance(self, time, state, action)

      Generate disturbance.


   .. py:method:: dynamics(self, time, state, action, disturbance)

      Dynamics for the system.
