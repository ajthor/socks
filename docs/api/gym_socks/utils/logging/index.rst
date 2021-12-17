:py:mod:`~gym_socks.utils.logging`
==================================

.. py:module:: gym_socks.utils.logging


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.utils.logging.ms_tqdm




.. py:class:: ms_tqdm(iterable=None, desc=None, total=None, leave=True, file=None, ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=None, lock_args=None, nrows=None, colour=None, delay=0, gui=False, **kwargs)

   Bases: :py:obj:`tqdm.auto.tqdm`

   A custom tqdm progress bar implementation.

   This is a simple modification to the tqdm progress bar that shows millisecond
   computation times in the progress bar. Adds the `elapsed_ms` format specifier to
   the format string.


   .. py:method:: format_dict(self)
      :property:

      Public API for read-only member access.
