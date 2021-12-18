:orphan:

:py:mod:`~examples._computation_timer`
======================================

.. py:module:: examples._computation_timer

.. autoapi-nested-parse::

   Computation timer.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   examples._computation_timer.ComputationTimer




Attributes
~~~~~~~~~~

.. autoapisummary::

   examples._computation_timer.logger


.. py:data:: logger




.. py:class:: ComputationTimer

   Bases: :py:obj:`object`

   Computation timer.

   Simple timer class for measuring the time of an algorithm and displaying the result.

   .. rubric:: Example

   >>> from examples._computation_timer import ComputationTimer
   >>> with ComputationTimer():
   ...     # run algorithm
   computation time: 3.14159 s

   .. py:method:: __enter__(self)


   .. py:method:: __exit__(self, *exc_info)


   .. py:method:: log_time(self)

      Output the computation time to the log.
