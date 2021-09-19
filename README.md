# Gym-SOCKS :socks:

Gym-SOCKS is a suite of algorithms for stochastic optimal control using kernel
methods that runs on top of [OpenAI Gym](https://gym.openai.com).

## Installation

To install the toolbox, use `pip install gym-socks`.

Alternatively, download the code from the GitHub repo and install using `pip install .` from the code directory.

We support Python versions 3.7, 3.8 and 3.9 on Linux and macOS. We do not officially support Windows.

## Examples

Gym-SOCKS comes with several examples in the GitHub repo. In order to run the examples, first install the package and use `python -m examples.<example>` from the code directory. Currently, the included examples are:

1. `kernel_sr`
2. `kernel_sr_max`
3. `kernel_control`
4. `monte_carlo_sr`

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
