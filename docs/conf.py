# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import shutil
import os
import sys

sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve
from github_link import _get_git_revision

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "SOCKS"
copyright = "2021, Adam Thorpe"
author = "Adam Thorpe"

# The full version, including alpha/beta/rc tags
release = "0.1.0-alpha0"

git_revision = _get_git_revision()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc.typehints",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

mathjax3_config = {
    "startup": {
        "requireMap": {
            "AMSmath": "ams",
            "AMSsymbols": "ams",
            "AMScd": "amscd",
            "HTML": "html",
            "noErrors": "noerrors",
            "noUndefined": "noundefined",
        }
    },
    "tex": {
        "tagSide": "right",
    },
}

autodoc_typehints = "description"

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "sklearn",
    "gym",
    "sacred",
    "tqdm",
]

# bibtex
bibtex_bibfiles = ["bibliography.bib"]

# Code block and signature options.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

linkcode_resolve = make_linkcode_resolve(
    "gym_socks",
    "https://github.com/ajthor/socks/blob/{revision}/{package}/{path}#L{lineno}",
)

# nbsphinx configuration options
nbsphinx_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}

shutil.copytree(
    os.path.join("..", "examples"),
    os.path.join("..", "docs/examples"),
    dirs_exist_ok=True,
)

# print(git_revision)
# https://mybinder.org/v2/gh/scikit-learn/scikit-learn/1.0.X?urlpath=lab/tree/notebooks/auto_examples/ensemble/plot_adaboost_regression.ipynb
# https://hub.gke2.mybinder.org/user/scikit-learn-scikit-learn-q3mk29qx/lab/tree/notebooks/auto_examples/ensemble/plot_adaboost_regression.ipynb

# https://scikit-learn.org/stable/_downloads/38e826c9e3778d7de78b2fc671fd7903/plot_adaboost_regression.ipynb

# https://mybinder.org/v2/gh/ajthor/socks/fa24fa3?filepath=/home/docs/checkouts/readthedocs.org/user_builds/socks/checkouts/feature-rework-sampling/docs/examples/identification/linear_id.ipynb

# https://mybinder.org/v2/gh/ajthor/socks/fa24fa3?urlpath=https://socks.readthedocs.io/en/feature-rework-sampling/examples/identification/linear_id.ipynb

# https://hub.gke2.mybinder.org/user/ajthor-socks-t8gkdjbk/https://socks.readthedocs.io/en/feature-rework-sampling/examples/identification/linear_id.ipynb

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        Open an interactive version of this example on Binder:
        :raw-html:`<a href="https://mybinder.org/v2/gh/ajthor/socks/{{ env.config.git_revision }}?filepath={{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:middle"></a>`

    __ https://github.com/ajthor/socks/blob/
        {{ env.config.git_revision }}/{{ docname }}

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ env.doc2path(env.docname, base='doc') | escape_latex }}}} ends here.}}
"""