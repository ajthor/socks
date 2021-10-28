"""Plotting ingredient.

This ingredient is used for experiments with a plotting component.

"""

from sacred import Ingredient

import numpy as np

plotting_ingredient = Ingredient("plot")


@plotting_ingredient.config
def _config():
    font_size = 8

    fig_height = 3  # Figure height in inches.
    fig_width = 3  # Figure width in inches.


@plotting_ingredient.capture
def load_matplotlib(font_size):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": font_size,
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    import matplotlib.pyplot as plt

    plt.set_loglevel("notset")

    return plt
