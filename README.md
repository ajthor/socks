# SOCKS :socks:

![build](https://github.com/ajthor/socks/actions/workflows/build.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ajthor/socks/HEAD)
[![Documentation Status](https://readthedocs.org/projects/socks/badge/?version=latest)](https://socks.readthedocs.io/en/latest/?badge=latest)

__NOTE__: SOCKS is still under active (alpha) development 🚧👷🚧. Code is provided
as-is, and some algorithms may not work as intended before the initial release. Please
check back soon for the official release.

SOCKS is a suite of algorithms for stochastic optimal control using kernel methods.

It runs alongside [OpenAI Gym](https://gym.openai.com), and comes with several classic
controls Gym environments. In addition, it can integrate with many pre-existing Gym
environments.

## Installation

To install the toolbox, use `pip install gym-socks`. Alternatively, download the code
from the GitHub repo and install using `pip install .` from the code directory.

We support Python versions 3.7, 3.8, and 3.9 on Linux and macOS. We do not officially
support Windows.

## Examples

SOCKS comes with several examples in the GitHub repo. In order to run the examples,
first install the package and use `python examples/<example>` from the code directory.

For example, `python python examples/control/tracking.py` will run the optimal control
algorithm on the tracking benchmark using nonholonomic vehicle dynamics.

## Citation

In order to cite the toolbox, use the following bibtex entry:

```
@inproceedings{thorpe2022hscc,
  title={{SOCKS}: A Kernel-Based Stochastic Optimal Control and Reachability Toolbox},
  authors={Thorpe, Adam J. and Oishi, Meeko M. K.},
  year={2022},
  booktitle={Proceedings of the 25th ACM International Conference on Hybrid Systems: Computation and Control},
}
```
