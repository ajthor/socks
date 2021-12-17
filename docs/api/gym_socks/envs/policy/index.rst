:py:mod:`~gym_socks.envs.policy`
================================

.. py:module:: gym_socks.envs.policy


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gym_socks.envs.policy.BasePolicy
   gym_socks.envs.policy.RandomizedPolicy
   gym_socks.envs.policy.ConstantPolicy
   gym_socks.envs.policy.ZeroPolicy




.. py:class:: BasePolicy

   Bases: :py:obj:`abc.ABC`

   Base policy class.

   This class is ABSTRACT, meaning it is not meant to be instantiated directly.
   Instead, define a new class that inherits from BasePolicy.

   The `__call__` method is the main point of entry for the policy classes. All
   subclasses must implement a `__call__` method. This makes the class callable, so
   that policies can be evaluated as::

       u = policy(x)

   .. note::

      Policies come in four main varieties:

      * Time-invariant open-loop policies.
      * Time-invariant closed-loop policies.
      * Time-varying open-loop policies.
      * Time-varying closed-loop policies.

      Thus, the arguments to the `__call__` method should allow for `time` and
      `state` to be specified (if needed), and should be optional kwargs, meaning
      they should have a `None` default value, like so:

          >>> def __call__(self, time=None, state=None):
          ...     ...

   .. py:attribute:: action_space




   .. py:attribute:: state_space




   .. py:method:: __call__(self, *args, **kwargs)
      :abstractmethod:

      Evaluate the policy.

      :returns: An action returned by the policy.
      :rtype: action



.. py:class:: RandomizedPolicy(action_space = None)

   Bases: :py:obj:`BasePolicy`

   Randomized policy.

   A policy which returns a random control action.

   :param system: The system the policy is defined on. Needed to specify the shape of
                  the inputs and outputs.

   .. py:method:: __call__(self, *args, **kwargs)

      Evaluate the policy.

      :returns: An action returned by the policy.
      :rtype: action



.. py:class:: ConstantPolicy(action_space = None, constant=0)

   Bases: :py:obj:`BasePolicy`

   Constant policy.

   A policy which returns a constant control action.

   :param system: The system the policy is defined on. Needed to specify the shape of
                  the inputs and outputs.
   :param constant: The constant value returned by the policy.

   .. py:method:: __call__(self, *args, **kwargs)

      Evaluate the policy.

      :returns: An action returned by the policy.
      :rtype: action



.. py:class:: ZeroPolicy(action_space = None)

   Bases: :py:obj:`ConstantPolicy`

   Zero policy.

   A policy which returns a constant (zero) control action.

   :param system: The system the policy is defined on. Needed to specify the shape of
                  the inputs and outputs.
