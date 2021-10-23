# SOCKS :socks:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

__NOTE__: SOCKS is still under active (alpha) development 🚧👷🚧. Code is provided as-is, and some algorithms may not work as intended before the initial release. Please check back soon for the official 1.0 release.

SOCKS is a suite of algorithms for stochastic optimal control using kernel
methods.

It runs on top of [OpenAI Gym](https://gym.openai.com), and comes with several classic controls Gym environments. In addition, it can integrate with many pre-existing Gym environments.

## Installation

To install the toolbox, use `pip install gym-socks`.

Alternatively, download the code from the GitHub repo and install using `pip install .` from the code directory.

We support Python versions 3.7, 3.8, and 3.9 on Linux and macOS. We do not officially support Windows.

## Examples

SOCKS comes with several examples in the GitHub repo. In order to run the examples, first install the package and use `python -m <example>` from the code directory.

For example, `python -m examples.nonholonomic.kernel_control_bwd` will run the nonholonomic vehicle (backward in time) optimal control algorithm.

## Citation

In order to cite the toolbox, use the following bibtex entry:

```
@inproceedings{thorpe2022hscc,
  title={{SOCKS}: A Kernel-Based Stochastic Optimal Control and Reachability Toolbox},
  authors={Thorpe, Adam J. and Oishi, Meeko M. K.},
  year={2022},
  booktitle={Proceedings of the 25th ACM International Conference on Hybrid Systems: Computation and Control (submitted)},
}
```
