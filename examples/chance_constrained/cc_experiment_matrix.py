import numpy as np

from examples.chance_constrained.benchmark_chance_constrained_problem import (
    ex as cc_experiment,
)
from examples.chance_constrained.benchmark_chance_constrained_problem_pd import (
    ex as cc_experiment_pd,
)

from examples.chance_constrained.cc_env import NonMarkovIntegratorEnv


def ex_matrix():

    sigmas = [
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
    # for i, sigma in enumerate(sigmas):
    #     print(f"Computing for sigma={sigma}")
    #     cc_experiment.run(
    #         config_updates={
    #             "seed": 0,
    #             "sigma": sigma,
    #             "sample": {"sample_space": {"sample_size": 500}},
    #             "delta": 0.01,
    #             "plot_cfg": {
    #                 "plot_filename": "results/plot_sigma{s:d}.png".format(s=i)
    #             },
    #         }
    #     )

    sample_sizes = [
        100,
        300,
        500,
        700,
        900,
        1100,
        1300,
        1500,
        1700,
        1900,
        2100,
        2300,
        2500,
    ]
    # for j, sample_size in enumerate(sample_sizes):
    #     print(f"Computing for sample_size={sample_size}")
    #     cc_experiment.run(
    #         config_updates={
    #             "seed": 0,
    #             "sigma": 5.0,
    #             "sample": {"sample_space": {"sample_size": int(sample_size)}},
    #             "delta": 0.01,
    #             "plot_cfg": {
    #                 "plot_filename": "results/plot_size{size:d}.png".format(size=j)
    #             },
    #         }
    #     )

    for i, sigma in enumerate(sigmas):
        for j, sample_size in enumerate(sample_sizes):
            print(f"Computing for sigma={sigma}, sample_size={sample_size}")
            filename = "results/matrix/plot_{sig:d}_{size:d}.png".format(sig=i, size=j)
            print(f"Saving as: {filename}")
            cc_experiment_pd.run(
                config_updates={
                    "seed": 0,
                    "sigma": sigma,
                    "sample": {"sample_space": {"sample_size": int(sample_size)}},
                    "delta": 0.01,
                    "plot_cfg": {"plot_filename": filename},
                }
            )


def ex_pd_vs_non():

    config_updates = {
        "seed": 0,
        "sigma": 5,
        "sample": {"sample_space": {"sample_size": 500}},
        "delta": 0.01,
    }

    print(f"Computing uniform (non-PD) solution.")
    filename = "results/plot_cc_un.png"
    print(f"Saving as: {filename}")
    cc_experiment_pd.run(
        config_updates={
            **config_updates,
            "plot_cfg": {"plot_filename": filename},
        }
    )

    print(f"Computing PD solution.")
    filename = "results/plot_cc_pd.png"
    print(f"Saving as: {filename}")
    cc_experiment_pd.run(
        config_updates={
            **config_updates,
            "plot_cfg": {"plot_filename": filename},
        }
    )


if __name__ == "__main__":
    # ex_matrix()
    # ex_pd_vs_non()

    test_system()
