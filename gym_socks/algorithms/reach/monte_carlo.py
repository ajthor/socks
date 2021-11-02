"""Stochastic reachability using Monte-Carlo."""

import gym
import gym_socks

import numpy as np

from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface
from gym_socks.algorithms.reach.reach_common import _tht_step, _fht_step

from gym_socks.envs.dynamical_system import DynamicalSystem
from gym_socks.envs.policy import BasePolicy
from gym_socks.envs.policy import RandomizedPolicy

from gym_socks.envs.sample import sample
from gym_socks.envs.sample import sample_generator

from gym_socks.utils import normalize, indicator_fn, generate_batches
from gym_socks.utils.logging import ms_tqdm, _progress_fmt
from tqdm.contrib.logging import logging_redirect_tqdm


def _trajectory_indicator(
    trajectories,
    num_steps=None,
    constraint_tube=None,
    target_tube=None,
    step_fn=None,
):
    trajectories = np.asarray(trajectories, dtype=np.float32)
    result = indicator_fn(trajectories[:, num_steps], target_tube[num_steps])
    result = np.array(result, dtype=int)

    for t in range(num_steps - 1, -1, -1):
        result = step_fn(trajectories[:, t], result, constraint_tube[t], target_tube[t])

    return result


def _monte_carlo_trajectory_sampler(
    env: DynamicalSystem = None,
    policy: BasePolicy = None,
    state: np.ndarray = None,
):
    """Default trajectory sampler.

    Args:
        env: The system to sample from.
        policy: The policy applied to the system during sampling.
        sample_space: The space where initial conditions are drawn from.

    Returns:
        A generator function that yields system observations as tuples.

    """

    @sample_generator
    def _sample_generator():

        state_sequence = []
        state_sequence.append(state)

        env.state = state

        time = 0
        for t in range(env.num_time_steps):
            action = policy(time=time, state=env.state)
            next_state, cost, done, _ = env.step(action)

            state_sequence.append(next_state)

            time += 1

        yield state_sequence

    return _sample_generator


def monte_carlo_sr(
    env: DynamicalSystem,
    policy: BasePolicy,
    T: np.ndarray,
    num_iterations: int = None,
    constraint_tube: list = None,
    target_tube: list = None,
    problem: str = "THT",
    verbose: bool = False,
):
    """Stochastic reachability using Monte-Carlo.

    Computes an approximation of the safety probabilities of the stochastic reachability
    problem using Monte-Carlo methods.

    Args:
        env: The dynamical system model. Needed to configure the sampling spaces.
        policy: The policy applied to the system during sampling.
        T: Points to estimate the safety probabilities at. Should be in the form of a
            2D-array, where each row indicates a point.
        num_iterations : Number of Monte-Carlo iterations.
        num_steps : Number of time steps to compute the approximation.
        constraint_tube : List of spaces or constraint functions. Must be the same
            length as `num_steps`.
        target_tube : List of spaces or target functions. Must be the same length as
            `num_steps`.
        problem : One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
            problem and `"FHT"` specifies the first-hitting time problem.
        verbose : Boolean flag to indicate verbose output.

    """

    alg = MonteCarloSR(
        num_iterations=num_iterations,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=problem,
        verbose=verbose,
    )
    return alg.fit_predict(env, policy, T)


class MonteCarloSR(AlgorithmInterface):
    """Stochastic reachability using Monte-Carlo.

    Computes an approximation of the safety probabilities of the stochastic reachability
    problem using Monte-Carlo methods.

    Args:
        num_iterations : Number of Monte-Carlo iterations.
        num_steps : Number of time steps to compute the approximation.
        constraint_tube : List of spaces or constraint functions. Must be the same
            length as `num_steps`.
        target_tube : List of spaces or target functions. Must be the same length as
            `num_steps`.
        problem : One of `{"THT", "FHT"}`. `"THT"` specifies the terminal-hitting time
            problem and `"FHT"` specifies the first-hitting time problem.
        verbose : Boolean flag to indicate verbose output.

    """

    def __init__(
        self,
        num_iterations: int = None,
        constraint_tube: list = None,
        target_tube: list = None,
        problem: str = "THT",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_iterations = num_iterations

        self.constraint_tube = constraint_tube
        self.target_tube = target_tube

        self.problem = problem

        self.verbose = verbose

    def _validate_params(self, S):

        if self.constraint_tube is None:
            raise ValueError("Must supply constraint tube.")

        if self.target_tube is None:
            raise ValueError("Must supply target tube.")

        if self.problem not in ("FHT", "THT"):
            raise ValueError(f"problem is not in {'THT', 'FHT'}")

        if self.problem == "THT":
            self.step_fn = _tht_step
        elif self.problem == "FHT":
            self.step_fn = _fht_step

    def _validate_data(self, S):

        if S is None:
            raise ValueError("Must supply a sample.")

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def fit_predict(self, env: DynamicalSystem, policy: BasePolicy, T: np.ndarray):
        """Run the algorithm.

        Computes the safety probabilities for the points provided.

        Args:
            env: The dynamical system model. Needed to configure the sampling spaces.
            T: Points to estimate the safety probabilities at. Should be in the form of
                a 2D-array, where each row indicates a point.

        Returns:
            The safety probabilities corresponding to each point. The output is in the
            form of a 2D-array, where each row corresponds to the points in `T` and the
            number of columns corresponds to the number of time steps.

        """

        self._validate_params(T)

        self._validate_data(T)
        num_test_points = len(T)
        T = np.array(T)

        pbar = ms_tqdm(
            total=num_test_points,
            bar_format=_progress_fmt,
            disable=False if self.verbose is True else True,
        )

        # Initialize the safety probability matrix.
        safety_probabilities = np.empty(
            (env.num_time_steps, num_test_points), dtype=np.float32
        )

        for i, state in enumerate(T):

            # For each point, generate a collection of trajectories.
            S = sample(
                sampler=_monte_carlo_trajectory_sampler(
                    env=env, state=state, policy=policy
                ),
                sample_size=self.num_iterations,
            )

            # Working backwards in time, compute the "likelihood" that the trajectories
            # will satisfy the constraints set up by the constraint tube and target
            # tube.
            for t in range(env.num_time_steps - 1, -1, -1):

                satisfies_constraints = _trajectory_indicator(
                    S,
                    num_steps=t,
                    constraint_tube=self.constraint_tube,
                    target_tube=self.target_tube,
                    step_fn=self.step_fn,
                )

                safety_probabilities[t, i] = (
                    satisfies_constraints.sum() / self.num_iterations
                )

            pbar.update()

        pbar.close()

        # Return the flipped safety probabilities to be in line with kernel SR.
        return safety_probabilities[::-1, ...]
