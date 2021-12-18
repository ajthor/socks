:py:mod:`~examples.ingredients.common`
======================================

.. py:module:: examples.ingredients.common

.. autoapi-nested-parse::

   Common utility functions for ingredients.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   examples.ingredients.common.assert_config_has_key
   examples.ingredients.common.parse_array
   examples.ingredients.common.box_factory
   examples.ingredients.common.grid_ranges
   examples.ingredients.common.grid_sample_size



.. py:function:: assert_config_has_key(_config, key)


.. py:function:: parse_array(value, shape, dtype)

   Utility function for parsing configuration variables.

   Parses values passed as either a scalar or list into an array of type `dtype`.

   :param value: The value input to the config.
   :param shape: The shape of the resulting array.
   :param dtype: The type to cast to.

   :returns: An array of a particular length.


.. py:function:: box_factory(lower_bound, upper_bound, shape, dtype)

   Box space factory.

   Creates a `gym.spaces.Box` to be used by sampling functions.



.. py:function:: grid_ranges(space, grid_resolution)

   Compute grid ranges.


.. py:function:: grid_sample_size(space, grid_resolution)

   Compute grid sample size.
