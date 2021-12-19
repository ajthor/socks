Policies
========

By definition, a policy is a function which returns a control action. In SOCKS, a policy
is defined as a callable class that inherits from
:py:class:`~gym_socks.envs.policy.BasePolicy`, which is also used as a parent class for
algorithms which compute control policies.

Thus, a policy simply inherits from :py:class:`~gym_socks.envs.policy.BasePolicy` and
implements a custom :py:meth:`~gym_socks.envs.policy.BasePolicy.__call__` function.

.. important::

    The main thing to know when defining custom policies is that functions in SOCKS
    which sample from systems pass all relevant information to the policy for the
    purpose of computing a control action. Since the policy can be either time-varying
    or time-invariant or open- or closed-lopp, it may require the time and state of the
    system in order to compute the control action. Thus, custom policies should
    explicitly name these variables as parameters if they are required, and should use
    ``*args, **kwargs`` to capture additional, unneeded arguments.

Templates
---------

It is recommended to use the following template when defining new policies to use in
algorithms.

Default Template
~~~~~~~~~~~~~~~~

.. code-block:: python

    from gym_socks.envs.policy import BasePolicy

    class CustomPolicy(BasePolicy):

        def __init__(self, action_space: Space = None):
            self.action_space = action_space

        def __call__(self, *args, **kwargs):
            return self.action_space.sample()