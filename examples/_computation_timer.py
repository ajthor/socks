"""Computation timer."""

import logging
from time import perf_counter

logger = logging.getLogger("timer")


class ComputationTimer(object):
    """Computation timer.

    Simple timer class for measuring the time of an algorithm and displaying the result.

    Example:

        >>> from examples._computation_timer import ComputationTimer
        >>> with ComputationTimer():
        ...     # run algorithm
        computation time: 3.14159 s

    """

    def __init__(self) -> None:
        self._start_time = None

    def __enter__(self):
        logger.debug("Starting timer.")
        self._start_time = perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.log_time()
        self._start_time = None

    def log_time(self):
        """Output the computation time to the log."""
        elapsed_time = perf_counter() - self._start_time
        logger.debug("Stopping timer.")
        logger.info(f"computation time: {elapsed_time} s")
