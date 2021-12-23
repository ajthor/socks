Creating New Examples
=====================

We welcome new examples that showcase the algorithms and techniques in SOCKS, including
special use cases that may be of interest to the wider community.

Approved examples will be included in subsequent releases of SOCKS, and will be featured
in the :doc:`/examples/index` section. In order to submit an example, please first open
an `issue <https://github.com/ajthor/socks/issues>`_ and write a short proposal of the
example that you would like to include, as well as a description of the example's
output. Then, open a pull request for your example so that we can work with you to
incorporate it into the docs and make sure that it follows the guidelines of the
codebase.

.. admonition:: Have a different technique that you would like to compare against SOCKS?

    We are also interested in developing benchmarks for comparison against other
    existing techniques. See the :doc:`/benchmarks/new_benchmarks` page for more
    information.

Main Points
-----------

We have integrated our documentation with `binder <https://mybinder.org>`_ to allow
users to open an interactive version of the examples and modify them. Our examples are
written as ``.py`` scripts that use comments to denote Markdown and code blocks that are
displayed nicely on the documentation page and can be imported easily into binder.
Adding these comment lines is relatively painless, and does not affect the execution of
the script if run via the terminal. Thus, the process of creating a new example should
be straightforward:

1. Put your example in a single script file in the ``/examples`` folder of the repo.
2. Make sure your script runs without any other external dependencies besides those
   provided in the `Dockerfile <https://github.com/ajthor/socks/blob/main/Dockerfile>`_.
3. Add ``# %%`` comments to indicate code blocks and ``# %% [markdown]`` blocks to
   document the code periodically. Markdown formatting will be rendered automatically,
   so be sure to include headers and a title line at the top of your example that will
   be used to generate the contents.


Formatting
~~~~~~~~~~

Examples are written using the "percent" `format`_ for Jupyter notebooks.

We use this style because it is compatible with several editors and IDEs, including:
`Hydrogen`_ for Atom, `VS Code`_, and `PyCharm`_, to name a few.

In a nutshell, code blocks are preceded by a ``# %%`` comment line, which indicates that
everything coming after is to be interpreted as code. Markdown blocks are preceded by
``# %% [markdown]``, and everything after (in comments) is written in Markdown and
converted to ``rst`` at compile-time. See `the Jupytext docs`__ or the
`nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_ docs for more formatting info.

__ format_

Here is a small example:

.. code-block:: python

    # %% [markdown]
    """
    Everything in this cell is rendered as markdown.
    """

    # %% [markdown]
    # Multiline markdown blocks
    # are also allowed.

    # %%
    # This is a code block. This line is rendered as a code comment.
    a = 1 + 2
    print(a)  # This code will be run.

The above example is rendered below.

----

Everything in this cell is rendered as markdown.

Multiline markdown blocks are also allowed.

.. nbinput:: ipython3
    :execution-count: 1

    # This is a code block. This line is rendered as a code comment.
    a = 1 + 2
    print(a)  # This code will be run.

.. nboutput::
    :execution-count: 1

    3

----

Have questions? Feel free to open an `issue <https://github.com/ajthor/socks/issues>`_.

.. _format: https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format

.. _Hydrogen: https://nteract.gitbooks.io/hydrogen/docs/Usage/NotebookFiles.html

.. _VS Code: https://code.visualstudio.com/docs/python/jupyter-support-py

.. _PyCharm: https://www.jetbrains.com/help/pycharm/editing-jupyter-notebook-files.html
