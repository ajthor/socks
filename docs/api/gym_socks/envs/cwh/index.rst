:py:mod:`~gym_socks.envs.cwh`
=============================

.. py:module:: gym_socks.envs.cwh


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.cwh.BaseCWH
   gym_socks.envs.cwh.CWH4DEnv
   gym_socks.envs.cwh.CWH6DEnv




.. py:class:: BaseCWH

   Bases: :py:obj:`object`

   CWH base class.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from `BaseCWH`, and define a custom
   `compute_state_matrix` and `compute_input_matrix` function.

   This class holds the shared parameters for the CWH systems, which include:

   * orbital radius
   * gravitational constant
   * celestial body mass
   * chief mass

   And provides methods to compute:

   * graviational parameter (mu)
   * angular velocity (n)


   .. py:attribute:: time_horizon
      :annotation: = 600



   .. py:attribute:: sampling_time
      :annotation: = 20



   .. py:method:: orbital_radius(self)
      :property:


   .. py:method:: gravitational_constant(self)
      :property:


   .. py:method:: celestial_body_mass(self)
      :property:


   .. py:method:: chief_mass(self)
      :property:


   .. py:method:: compute_mu(self)


   .. py:method:: compute_angular_velocity(self)


   .. py:method:: compute_state_matrix(self, sampling_time)
      :abstractmethod:


   .. py:method:: compute_input_matrix(self, sampling_time)
      :abstractmethod:



.. py:class:: CWH4DEnv(seed=None, *args, **kwargs)

   Bases: :py:obj:`BaseCWH`, :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   4D Clohessy-Wiltshire-Hill (CWH) system.

   The 4D CWH system is a simplification of the 6D dynamics to operate within a plane.
   Essentially, it ignores the 'z' component of the dynamics.


   .. py:method:: compute_state_matrix(self, sampling_time)


   .. py:method:: compute_input_matrix(self, sampling_time)


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

      NOTE: The CWH system has a closed-form solution for the equations of
      motion, meaning the dynamics function presented here is primarily for
      reference. The scipy.solve_ivp function does not return the correct
      result for the dynamical equations, and will quickly run into numerical
      issues where the states explode. See the 'step' function for details
      regarding how the next state is calculated.



.. py:class:: CWH6DEnv(seed=None, *args, **kwargs)

   Bases: :py:obj:`BaseCWH`, :py:obj:`gym_socks.envs.dynamical_system.DynamicalSystem`

   6D Clohessy-Wiltshire-Hill (CWH) system.

   .. py:method:: compute_state_matrix(self, sampling_time)


   .. py:method:: compute_input_matrix(self, sampling_time)


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

      NOTE: The CWH system has a closed-form solution for the equations of
      motion, meaning the dynamics function presented here is primarily for
      reference. The scipy.solve_ivp function does not return the correct
      result for the dynamical equations, and will quickly run into numerical
      issues where the states explode. See the 'step' function for details
      regarding how the next state is calculated.
