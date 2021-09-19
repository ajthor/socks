# from algorithms.algorithm import AlgorithmRunner
from gym_socks.algorithms.control.kernel_control.kernel_control import KernelControlFwd

import gym
import gym_socks

import numpy as np

import gym_socks.kernel.metrics as kernel
from gym_socks.envs.sample import generate_sample
from gym_socks.envs.sample import generate_uniform_sample

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt


def main():

    env = gym_socks.envs.StochasticNDIntegratorEnv(2)

    num_time_steps = env.num_time_steps

    constraint_tube = [
        gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        for i in range(num_time_steps)
    ]

    sample_space = gym.spaces.Box(
        low=-1.1,
        high=1.1,
        shape=env.observation_space.shape,
        dtype=np.float32,
    )

    # S = generate_sample(sample_space, env, 5000)
    S, _ = generate_uniform_sample(sample_space=sample_space, system=env, n=[71, 71])

    # rounding to avoid numpy floating point precision errors
    x1 = np.round(np.linspace(-1, 1, 81), 3)
    x2 = np.round(np.linspace(-1, 1, 81), 3)
    T = [(xx1, xx2) for xx1 in x1 for xx2 in x2]
    # _, T = generate_uniform_sample(sample_space, env, [21, 21])

    alg = KernelSR()

    Pr, _ = alg.run(env=env, sample=S, test_points=T, constraint_tube=constraint_tube)

    fig = plt.figure()
    cm = "viridis"

    # ax = fig.add_subplot(111)
    #
    # XX, YY = np.meshgrid(x1, x2, indexing="ij")
    # Z = Pr[5].reshape(XX.shape)
    #
    # plt.pcolor(XX, YY, Z, cmap=cm, vmin=0, vmax=1)
    # plt.colorbar()
    #
    # plt.savefig("plot.png", dpi=300, bbox_inches="tight")

    ax = fig.add_subplot(111, projection="3d")

    XX, YY = np.meshgrid(x1, x2, indexing="ij")
    Z = Pr[0].reshape(XX.shape)

    ax.plot_surface(XX, YY, Z, cmap=cm, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$\Pr$")
    # ax.view_init(elev=90., azim=0.)

    plt.savefig("results/plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
