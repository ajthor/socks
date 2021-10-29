"""Plotting ingredient.

This ingredient is used for experiments with a plotting component.

"""

import os

import numpy as np

from sacred import Ingredient

plotting_ingredient = Ingredient("plot_cfg")


@plotting_ingredient.config
def _config():
    rc_params = dict()
    rc_params_filename = os.path.abspath("examples/ingredients/matplotlibrc")

    plot_filename = "results/plot.png"


@plotting_ingredient.capture
def update_rc_params(matplotlib, rc_params, rc_params_filename):
    custom_rc_params = matplotlib.rc_params_from_file(
        fname=rc_params_filename, use_default_template=True
    )
    matplotlib.rcParams.update(custom_rc_params)

    for key, value in rc_params.items():
        matplotlib.rc(key, **value)
