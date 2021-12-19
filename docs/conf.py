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

import os
import sys

sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "SOCKS"
copyright = "2021, Adam Thorpe"
author = "Adam Thorpe"

# The full version, including alpha/beta/rc tags
release = "0.1.0-alpha0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
    "sphinx.ext.autodoc.typehints",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
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

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autodoc_typehints = "description"

autosummary_generate = False

autoapi_root = "api"
autoapi_dirs = ["../gym_socks", "../examples"]
autoapi_ignore = ["*tests*"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

autoapi_generate_api_docs = False
autoapi_keep_files = True

bibtex_bibfiles = ["bibliography.bib"]

linkcode_resolve = make_linkcode_resolve(
    "gym_socks",
    "https://github.com/ajthor/socks/blob/main/{package}/{path}#L{lineno}",
)
