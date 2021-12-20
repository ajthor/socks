"""SOCKS - A kernel-based stochastic optimal control toolbox."""


import logging

__all__ = ["algorithms", "kernel", "systems", "utils"]

logger = logging.getLogger(__name__)
"""Default logger for gym_socks.

Used mainly for debugging. Can be diabled entirely by setting the log level of the
logger to "notset".

"""

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
