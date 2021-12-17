:py:mod:`~gym_socks.envs.core`
==============================

.. py:module:: gym_socks.envs.core


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.core.BaseDynamicalObject




.. py:class:: BaseDynamicalObject

   Bases: :py:obj:`gym.Env`, :py:obj:`abc.ABC`

   Base dynamical object class.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from BaseDynamicalObject.

   This class serves as the base interface for dynamical objects, represented in most
   cases by either a `DynamicalSystem` or an obstacle.


   .. py:method:: step(self)
      :abstractmethod:

      Advance the object forward in time.

      Advances the object in the simulation. By default, an object is uncontrolled,
      meaning it accepts no parameters and the system evolves forward in time
      according to its own, internal dynamics. Controlled systems should accept an
      `action` parameter, which represents the user-selected control input.

      Additionally, time-varying systems should also accept a `time` parameter.



   .. py:method:: reset(self)
      :abstractmethod:

      Reset the object to an initial state.

      Resets the object to an initial state (which may be random).

      :returns: The new initial state of the object.


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


   .. py:method:: seed(self, seed=None)

      Sets the seed of the random number generator.

      This is primarily useful for objects which incorporate some sort of
      stochasticity to ensure repeatability.

      :param seed: Integer value representing the random seed.

      :returns: The seed of the RNG.
