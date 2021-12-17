:py:mod:`~gym_socks.envs.world`
===============================

.. py:module:: gym_socks.envs.world


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.world.World




.. py:class:: World

   Bases: :py:obj:`collections.abc.MutableSequence`

   World.

   The `World` class is essentially a `MutableSequence` (such as a list), of objects
   contained within the world. The objects contained within the world must be of type
   `_WorldObject`, meaning the implement the `step`, `reset`, `render`, `close`, and
   `seed` methods. If an object that does not implement these methods is added, an
   assertion error will be thrown.

   It can be used in much the same was as a list, for example::

       >>> world = World()
       >>> world.append(item)
       >>> world[1] = item
       >>> world += [item]

   The `World` is primarily used for keeping track of multiple objects, such as
   obstacles, which are contained within the world environment and simulated together.
   While it can track `DynamicalSystem`s, it is moreso intended to track uncontrolled
   or fully autonomous systems or objects, and to synchronize the simulation time.

   In addition, it implements the following methods, which are applied to all objects
   in the world environment:
   * `step`
   * `reset`
   * `render`
   * `close`
   * `seed`

   E.g., the `step` method calls the `step` method for each item in the world. The
   order in which the items are iterated over is the same as the order of the list. If
   the functions return a value, the results are given in a list.


   .. py:method:: time_horizon(self)
      :property:

      Time horizon for the simulation.


   .. py:method:: __getitem__(self, _index)


   .. py:method:: __setitem__(self, _index, _object)


   .. py:method:: __delitem__(self, _index)


   .. py:method:: __len__(self)


   .. py:method:: insert(self, _index, _object)

      S.insert(index, value) -- insert value before index


   .. py:method:: step(self, time = None)

      Advances the simulation forward one time step.

      :param time: The simulation time.

      :returns: The result of each step function in a list.


   .. py:method:: reset(self)

      Reset the world to a random initial condition.

      :returns: The result of each reset function in a list.


   .. py:method:: render(self, mode = 'human')


   .. py:method:: close(self)


   .. py:method:: seed(self, seed = None)

      Sets the seed of the random number generator.

      This is primarily useful for objects which incorporate some sort of
      stochasticity to ensure repeatability.

      :param seed: Integer value representing the random seed.

      :returns: The seed of the RNG for each object in a list.
