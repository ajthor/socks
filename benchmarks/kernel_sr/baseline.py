import numpy as np

from sacred import Experiment
from sacred import Ingredient

from time import perf_counter
from functools import lru_cache
from functools import partial

from gym.spaces import Box

from gym_socks.envs.integrator import NDIntegratorEnv

from gym_socks.sampling import random_sampler
from gym_socks.sampling import sample
from gym_socks.sampling import default_sampler

from gym_socks.algorithms.reach import kernel_sr

from sklearn.metrics.pairwise import rbf_kernel


ex = Experiment()


@ex.config
def ex_config():
    dimensionality = 2
    time_horizon = 10
    sample_size = 1000
    test_sample_size = 1000
    problem = "THT"
    regularization_param = 1
    sigma = 0.1


@lru_cache(maxsize=1)
@ex.capture
def generate_dataset(env, sample_size):
    """Generate the sample (dataset) for the experiment."""

    sample_space = Box(
        low=-1.1,
        high=1.1,
        shape=env.state_space.shape,
        dtype=env.state_space.dtype,
    )

    state_sampler = random_sampler(sample_space=sample_space)
    action_sampler = random_sampler(sample_space=env.action_space)

    S = sample(
        sampler=default_sampler(
            state_sampler=state_sampler, action_sampler=action_sampler, env=env
        ),
        sample_size=sample_size,
    )

    return S


@lru_cache(maxsize=1)
@ex.capture
def generate_test_dataset(env, test_sample_size):
    """Generate the test dataset."""

    sample_space = Box(
        low=-1,
        high=1,
        shape=env.state_space.shape,
        dtype=env.state_space.dtype,
    )

    T = sample(
        sampler=random_sampler,
        sample_size=test_sample_size,
        sample_space=sample_space,
    )

    return T


def generate_constraint_tube(env, time_horizon):
    """Generate the constraint tube."""

    constraint_tube = [
        Box(
            low=-1,
            high=1,
            shape=env.state_space.shape,
            dtype=env.state_space.dtype,
        )
    ] * time_horizon

    return constraint_tube


def generate_target_tube(env, time_horizon):
    """Generate the target tube."""

    target_tube = [
        Box(
            low=-0.5,
            high=0.5,
            shape=env.state_space.shape,
            dtype=env.state_space.dtype,
        )
    ] * time_horizon

    return target_tube


@ex.main
def main(
    dimensionality,
    time_horizon,
    sample_size,
    test_sample_size,
    problem,
    regularization_param,
    sigma,
):
    env = NDIntegratorEnv(dim=dimensionality)
    S = generate_dataset(env, sample_size)
    T = generate_test_dataset(env, test_sample_size)

    start_time = perf_counter()

    # Algorithm setup.
    constraint_tube = generate_constraint_tube(env, time_horizon)
    target_tube = generate_target_tube(env, time_horizon)

    # Main algorithm.
    kernel_sr(
        S=S,
        T=T,
        time_horizon=time_horizon,
        constraint_tube=constraint_tube,
        target_tube=target_tube,
        problem=problem,
        regularization_param=regularization_param,
        kernel_fn=partial(rbf_kernel, gamma=1 / (2 * (sigma ** 2))),
        verbose=False,
    )

    elapsed_time = perf_counter() - start_time
    return elapsed_time


if __name__ == "__main__":
    ex.run_commandline()
