:py:mod:`~gym_socks.envs.integrator`
====================================

.. py:module:: gym_socks.envs.integrator

.. autoapi-nested-parse::

   ND Integrator system.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.integrator.NDIntegratorEnv




.. .. autoclass:: gym_socks.envs.integrator.NDIntegratorEnv
.. py:class:: NDIntegratorEnv(dim = 1, seed=None, *args, **kwargs)

   Bases: :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   ND integrator system.

   An integrator system is an extremely simple dynamical system model, typically used
   to model a single variable and its higher order derivatives, where the input is
   applied to the highest derivative term, and is "integrated" upwards.

   A 2D Integrator system, for example, corresponds to the position and velocity
   components of a variable, where the input is applied to the velocity and then
   integrates upward to the position variable. Chaining two 2D integrator systems can
   model a system with x/y position and velocity.

   :param dim: The dimension of the integrator system.

   .. rubric:: Example

   >>> from gym_socks.envs import NDIntegratorEnv
   >>> env = NDIntegratorEnv(dim=2)
   >>> env.reset()
   >>> num_steps = env.num_time_steps
   >>> for i in range(num_steps):
   ...     action = env.action_space.sample()
   ...     obs, reward, done, _ = env.step(action)
   >>> env.close()

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
