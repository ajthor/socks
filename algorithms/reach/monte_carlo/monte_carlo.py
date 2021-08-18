from functools import partial

from algorithms.algorithm import AlgorithmInterface

import kernel.metrics as kernel
from systems.sample import generate_sample_trajectories

import gym

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel


def normalize(v):
    return v / np.sum(v, axis=0)


class MonteCarloSR(AlgorithmInterface):
    """
    Stochastic reachability using Monte-Carlo sampling.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the algorithm.
        """
        super().__init__(*args, **kwargs)

        # Global algorithm parameters go here.

    def run(
        self, system=None, test_points=None, constraint_tube=None, num_mc_samples=None
    ):

        if system is None:
            print("Must supply a system.")

        if test_points is None:
            print("Must supply test points.")

        if constraint_tube is None:
            print("Must supply a constrint tube.")

        if num_mc_samples is None:
            num_mc_samples = 100

        Xt = np.array(test_points)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(test_points)

        Pr = np.zeros((num_test_points, num_time_steps))

        def generate_state_trajectory(x0):
            system.state = x0

            def generate_next_state():
                action = system.action_space.sample()
                next_state, reward, done, _ = system.step(action)
                return next_state

            return np.array(
                [generate_next_state() for i in range(system.num_time_steps)]
            )

        for i, point in enumerate(test_points):

            S = np.array(
                [generate_state_trajectory(point) for j in range(num_mc_samples)]
            )

            # print(S)

            in_safe_set = np.array(
                [
                    [
                        constraint_tube[j].contains(np.array(trajectory[j]))
                        for j in range(num_time_steps)
                    ]
                    for trajectory in S
                ],
                dtype=np.float32,
            )

            # print(in_safe_set)
            # print(np.sum(in_safe_set, axis=0))
            #
            # print(Pr)
            Pr[i, :] = np.sum(in_safe_set, axis=0) / num_mc_samples

        return np.flipud(Pr.T)
