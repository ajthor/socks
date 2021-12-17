:py:mod:`~gym_socks.envs.nonholonomic`
======================================

.. py:module:: gym_socks.envs.nonholonomic

.. autoapi-nested-parse::

   Nonholonomic vehicle system.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.nonholonomic.NonholonomicVehicleEnv




.. py:class:: NonholonomicVehicleEnv(seed=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   Nonholonomic vehicle system.

   A nonholonomic vehicle (car-like) is typically modeled using what are known as
   "unicycle" dynamics. It is useful for modeling vehicles which can move forward and
   backward, and incorporates a steering angle or heading. The inputs are the velocity
   and change in steering angle.


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

      The dynamics are typically specified by a function::
          y = f(t, x, u, w)
                ┬  ┬  ┬  ┬
                │  │  │  └┤ w : Disturbance
                │  │  └───┤ u : Control action
                │  └──────┤ x : System state
                └─────────┤ t : Time variable
