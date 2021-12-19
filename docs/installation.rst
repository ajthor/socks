Installation
============

Installing the Latest Release
-----------------------------

SOCKS requires python 3.7 or greater to work.

The toolbox currently supports linux and macOS systems, but we do not officially support
Windows. The code should still work on Windows systems, but this is currently untested.

.. tab:: pip

    In order to avoid conflicts with other packages, it is highly recommended to use a
    virtual environment.

    .. tab:: Linux

        Install ``python3`` using the package manager.

        For instance, on Ubuntu 20.04, ensure that ``python3`` is installed on your system using ``python3 --version``, and make sure that the version is greater than 3.7. Then, to install ``pip``, use:

        .. code-block:: shell

            apt update & apt install python3-pip

    .. tab:: MacOS

        Install ``python3`` via homebrew ``brew install python`` or from
        `<http://python.org>`_.

    .. tab:: Windows

        Install ``python3``, for instance from `<http://python.org>`_.

    Then, run the following command:

    .. code-block:: shell

        pip install gym-socks

    Alternatively, to install from source, download the latest code from `GitHub
    <https://github.com/ajthor/socks/>`_, and from the code directory, run the following command:

    .. code-block:: shell

        pip install .

.. tab:: docker

    A `Dockerfile <https://github.com/ajthor/socks/blob/main/Dockerfile>`_ is provided
    with the repo which installs all necessary dependencies and installs SOCKS
    automatically. First, install `Docker <https://docs.docker.com/get-docker/>`_ on
    your system. Then, to build the docker image from the Dockerfile, use the following
    command:

    .. code-block:: shell

        docker build -t socks --build-arg PYTHON_VERSION:3.9 .

    Then, to launch the ``socks`` docker container, use:

    .. code-block:: shell

        docker run -it socks bash

    We use this docker container for all development and testing.


Troubleshooting
---------------

If you are having issues installing the code, open an issue at
`<https://github.com/ajthor/socks/issues>`_ and describe your issue, including all
relevant error messages.

Common Issues
~~~~~~~~~~~~~

* SOCKS currently depends on ``scikit-learn``, ``numpy``, and ``scipy`` to work. In our
  experience, installing these packages via linux package managers can lead to
  difficult to diagnose errors in the package setup. For instance, on Ubuntu, we
  recommend installing ``libblas-dev`` and ``liblapack-dev`` using ``apt``, and then
  installing ``scikit-learn``, ``numpy``, and ``scipy`` via pip.

  .. code-block:: shell

      apt update & apt install libblas-dev liblapack-dev
      pip install numpy scipy scikit-learn gym-socks