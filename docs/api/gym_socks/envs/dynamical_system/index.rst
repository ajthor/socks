:py:mod:`~gym_socks.envs.dynamical_system`
==========================================

.. py:module:: gym_socks.envs.dynamical_system


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.dynamical_system.DynamicalSystem




.. py:class:: DynamicalSystem

   Bases: :py:obj:`gym_socks.envs.core.BaseDynamicalObject`, :py:obj:`abc.ABC`

   Base class for dynamical system models.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from `DynamicalSystem`, and define a
   custom `dynamics` function.

   .. rubric:: Example

   >>> from gym_socks.envs.dynamical_system import DynamicalSystem
   >>> class CustomDynamicalSystem(DynamicalSystem):
   ...     def __init__(self):
   ...         super().__init__(state_dim=2, action_dim=1)
   ...
   ...     def dynamics(self, t, x, u, w):
   ...         return u

   .. note::

      * The state space and input space are assumed to be R^n and R^m, where n and m
      are set by state_dim and action_dim above (though these can be altered, see
      gym.spaces for more info).
      * The dynamics function (defined by you) returns dx/dt, and the system is
      integrated using scipy.integrate.solve_ivp to determine the state at the next
      time instant and discretize the system.
      * The reset function sets a new initial condition for the system.

   The system can then be simulated using the standard gym environment.

   .. rubric:: Example

   >>> import gym
   >>> import numpy as np
   >>> env = CustomDynamicalSystem()
   >>> env.reset()
   >>> num_steps = env.num_time_steps
   >>> for i in range(num_steps):
   ...     action = env.action_space.sample()
   ...     obs, reward, done, _ = env.step(action)
   >>> env.close()

   :param observation_space: The space of system observations.
   :param state_space: The state space of the system.
   :param action_space: The action (input) space of the system.
   :param seed: Random seed.
   :param euler: Whether to use the Euler approximation for simulation.

   .. py:attribute:: observation_space




   .. py:attribute:: action_space




   .. py:attribute:: state_space




   .. py:method:: sampling_time(self)
      :property:

      Sampling time.


   .. py:method:: generate_disturbance(self, time, state, action)

      Generate disturbance.


   .. py:method:: dynamics(self, time, state, action, disturbance)
      :abstractmethod:

      Dynamics for the system.

      The dynamics are typically specified by a function::
          y = f(t, x, u, w)
                ┬  ┬  ┬  ┬
                │  │  │  └┤ w : Disturbance
                │  │  └───┤ u : Control action
                │  └──────┤ x : System state
                └─────────┤ t : Time variable



   .. py:method:: generate_observation(self, time, state, action)

      Generate observation.


   .. py:method:: cost(self, time, state, action)

      Cost function for the system.


   .. py:method:: step(self, time=0, action=None)

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


   .. py:method:: reset(self)

      Reset the system to a random initial condition.


   .. py:method:: render(self, mode='human')
      :abstractmethod:

      Renders the environment.

      The set of supported modes varies per environment. (And some
      environments do not support rendering at all.) By convention,
      if mode is:

      - human: render to the current display or terminal and
        return nothing. Usually for human consumption.
      - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable
        for turning into a video.
      - ansi: Return a string (str) or StringIO.StringIO containing a
        terminal-style text representation. The text can include newlines
        and ANSI escape sequences (e.g. for colors).

      .. note::

         Make sure that your class's metadata 'render.modes' key includes
           the list of supported modes. It's recommended to call super()
           in implementations to use the functionality of this method.

      :param mode: the mode to render with
      :type mode: str

      Example:

      class MyEnv(Env):
          metadata = {'render.modes': ['human', 'rgb_array']}

          def render(self, mode='human'):
              if mode == 'rgb_array':
                  return np.array(...) # return RGB frame suitable for video
              elif mode == 'human':
                  ... # pop up a window and render
              else:
                  super(MyEnv, self).render(mode=mode) # just raise an exception


   .. py:method:: close(self)

      Override close in your subclass to perform any necessary cleanup.

      Environments will automatically close() themselves when
      garbage collected or when the program exits.


   .. py:method:: np_random(self)
      :property:

      Random number generator.


   .. py:method:: seed(self, seed=None)

      Sets the seed of the random number generator.

      :param seed: Integer value representing the random seed.

      :returns: The seed of the RNG.
