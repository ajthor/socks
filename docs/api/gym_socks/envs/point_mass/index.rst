:py:mod:`~gym_socks.envs.point_mass`
====================================

.. py:module:: gym_socks.envs.point_mass

.. autoapi-nested-parse::

   ND point mass system.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.point_mass.NDPointMassEnv




.. py:class:: NDPointMassEnv(dim, seed=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   ND point mass system.

   A point mass is a very simple system in which the inputs apply directly to the state
   variables. Thus, it is essentially a representation of a particle using Newton's
   equations `F = mA`.

   :param dim: Dimensionality of the point mass system.
   :param mass: Mass of the particle.

   .. py:method:: mass(self)
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
