:py:mod:`~gym_socks.algorithms.algorithm`
=========================================

.. py:module:: gym_socks.algorithms.algorithm


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.algorithms.algorithm.AlgorithmInterface




.. py:class:: AlgorithmInterface

   Bases: :py:obj:`abc.ABC`

   Base class for algorithms.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from AlgorithmInterface.

   The AlgorithmInterface is meant to mimic the sklearn estimator base class in order
   to promote a standard interface among machine learning algorithms. It requires that
   all subclasses implement a `fit` and `predict` method.


   .. py:method:: fit(cls, *args, **kwargs)
      :abstractmethod:


   .. py:method:: predict(cls, *args, **kwargs)
      :abstractmethod:
