from functools import partial

from gym_socks.algorithms.algorithm import AlgorithmInterface

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import sample_trajectories

import gym

import numpy as np


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
        self,
        system=None,
        T=None,
        policy=None,
        constraint_tube=None,
        target_tube=None,
        problem: "Stochastic reachability problem." = "THT",
        num_iterations=None,
    ):

        if system is None:
            print("Must supply a system.")

        if T is None:
            print("Must supply test points.")

        if constraint_tube is None:
            print("Must supply a constrint tube.")

        if target_tube is None:
            print("Must supply target tube.")
            return None

        if num_iterations is None:
            num_iterations = 100

        T = np.array(T)
        num_time_steps = system.num_time_steps - 1
        num_test_points = len(T)

        Pr = np.zeros((num_time_steps + 1, len(T)))

        tt_low = target_tube[num_time_steps].low
        tt_high = target_tube[num_time_steps].high

        for i in range(num_iterations):

            print(f"Computing iteration={i}")

            S, _ = sample_trajectories(system=system, initial_conditions=T)

            S = np.flip(S, 1)

            Pr[num_time_steps, :] += np.array(
                np.all(S[:, num_time_steps, :] >= tt_low, axis=1)
                & np.all(S[:, num_time_steps, :] <= tt_high, axis=1),
                dtype=np.float32,
            )

            for t in range(num_time_steps - 1, -1, -1):

                ct_low = constraint_tube[t].low
                ct_high = constraint_tube[t].high

                S_in_safe_set = np.all(S[:, t, :] >= ct_low, axis=1) & np.all(
                    S[:, t, :] <= ct_high, axis=1
                )

                if problem == "THT":

                    Pr[t, :] += np.array(S_in_safe_set, dtype=np.float32)

                elif problem == "FHT":

                    tt_low = target_tube[t].low
                    tt_high = target_tube[t].high

                    S_in_target_set = np.all(S[:, t, :] >= tt_low, axis=1) & np.all(
                        S[:, t, :] <= tt_high, axis=1
                    )

                    Pr[t, :] += np.array(S_in_target_set, dtype=np.float32) + np.array(
                        S_in_safe_set & ~np.array(S_in_target_set, dtype=bool),
                        dtype=np.float32,
                    )

        Pr /= num_iterations

        return Pr
