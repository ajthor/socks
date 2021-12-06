from abc import ABC, abstractmethod

import gym

from gym_socks.envs.dynamical_system import DynamicalSystem

import numpy as np


class BasePolicy(ABC):
    """Base policy class.

    This class is ABSTRACT, meaning it is not meant to be instantiated directly.
    Instead, define a new class that inherits from BasePolicy.

    The `__call__` method is the main point of entry for the policy classes. All
    subclasses must implement a `__call__` method. This makes the class callable, so
    that policies can be evaluated as::

        u = policy(x)

    Note:
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

    """

    action_space = None
    state_space = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Evaluate the policy.

        Returns:
            action: An action returned by the policy.

        """
        raise NotImplementedError


class RandomizedPolicy(BasePolicy):
    """Randomized policy.

    A policy which returns a random control action.

    Args:
        system: The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.

    """

    def __init__(self, action_space: gym.Space = None):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

    def __call__(self, *args, **kwargs):
        return self.action_space.sample()


class ConstantPresampledPolicy(BasePolicy):
    """Constant policy.

    A policy which returns a constant control action.

    Args:
        system: The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.
        constant: The constant value returned by the policy.

    """

    def __init__(self, controls, action_space=None):
        # controls is a list
        print("[ConstantPresampledPolicy] len(controls) = ", 
                len(controls))
        print("[ConstantPresampledPolicy] controls[0].shape = ", 
                controls[0].shape)
        self.action_space = action_space
        self._controls = controls
        self._nb_controls = len(controls)
        self._u_dim = self.action_space.shape[0]
        self._time_max = int(len(controls[0])/self._u_dim)
        self._id_sample = -1

    def __call__(self, time=0, *args, **kwargs):
        if time == 0:
            self._id_sample += 1
        if self._id_sample == self._nb_controls:
            self._id_sample = 0
        control_traj = self._controls[self._id_sample]
        control_traj = np.reshape(control_traj,
                                  (self._time_max, self._u_dim))
        control = control_traj[time,:]
        return np.full(
            self.action_space.shape,
            control,
            dtype=self.action_space.dtype,
        )


class ConstantPolicy(BasePolicy):
    """Constant policy.

    A policy which returns a constant control action.

    Args:
        system: The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.
        constant: The constant value returned by the policy.

    """

    def __init__(self, action_space: gym.Space = None, constant=0):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

        self._constant = constant

    def __call__(self, *args, **kwargs):
        return np.full(
            self.action_space.shape,
            self._constant,
            dtype=self.action_space.dtype,
        )


class ZeroPolicy(ConstantPolicy):
    """Zero policy.

    A policy which returns a constant (zero) control action.

    Args:
        system : The system the policy is defined on. Needed to specify the shape of
            the inputs and outputs.

    """

    def __init__(self, action_space: gym.Space = None):
        if action_space is not None:
            self.action_space = action_space
        else:
            raise ValueError("action space must be provided")

        self._constant = 0


class PDController(ConstantPolicy):
    """PD controller.

    Args:
        action_space : The action space of the policy.
        state_space : The state space of the system.
        goal_state : The goal state of the system.
        pd_gains : The gains of the controller.

    """

    def __init__(
        self,
        action_space: gym.Space = None,
        state_space: gym.Space = None,
        goal_state=np.zeros(4),
        PD_gains=np.zeros(4),
    ):
        if goal_state.shape != state_space.shape:
            raise ValueError(
                "goal_state should have same shape " + "as environment state"
            )
        if PD_gains.shape[0] != action_space.shape[0]:
            raise ValueError(
                "PD_gains[:,0] should have same shape " + "as environment action"
            )
        if PD_gains.shape[1] != state_space.shape[0]:
            raise ValueError(
                "PD_gains[0,:] should have same shape " + "as environment state"
            )

        self.state_space = state_space
        self.action_space = action_space
        self.goal_state = goal_state
        self.PD_gains = -1 * PD_gains

    def __call__(self, time=0, state=np.zeros(4), *args, **kwargs):
        control = np.zeros(self.action_space.shape, dtype=np.float32)

        # PD regulation to goal
        if time > 3:
            control = self.PD_gains @ (state - self.goal_state)

        # add exploratory noise
        sampled_control = self.action_space.sample()
        if time <= 3:
            control = control + 0.5*np.abs(sampled_control)

        # saturate control
        control = np.clip(control, self.action_space.low, self.action_space.high)

        return control.astype(
            self.action_space.dtype,
        )
