:py:mod:`~gym_socks.envs.tora`
==============================

.. py:module:: gym_socks.envs.tora

.. autoapi-nested-parse::

   TORA (translational oscillation with rotational actuation) system.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.tora.TORAEnv




.. py:class:: TORAEnv(seed=None)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   TORA (translational oscillation with rotational actuation) system.

   The TORA system is a mass with an attached pendulum (rotational oscillator) attached via a spring to a surface. This system is useful for modeling a variant of the pendulum system or a cart-pole system. The input is to the pendulum.


   .. py:method:: damping_coefficient(self)
      :property:


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
