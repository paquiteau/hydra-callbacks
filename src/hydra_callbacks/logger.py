"""Performance tracking tool."""

from time import perf_counter
from logging import Logger
import logging
from typing import Callable


class PerfLogger:
    """
    Simple Performance logger to use as a context manager.

    Parameters
    ----------
    logger
        A logger object or a callable
    level
        log level, default=0 , infos
    time_format
        format string to display the elapsed time

    Example
    -------
    >>> with PerfLogger(print) as pfl:
            time.sleep(1)
    """

    timers: dict[str, float] = {}
    timers_stack: list[str] = []
    _stop_time: float
    _start_time: float

    def __init__(
        self,
        logger: Logger | Callable,
        level: int = logging.INFO,
        name: str = "default",
        time_format: str = "{name} duration: {:.2f}s",
    ):
        self._name = name
        self._log_level = level

        if isinstance(logger, Logger):
            self.logger = lambda msg: logger.log(self._log_level, msg)
        elif callable(logger):
            self.logger = logger
        else:
            raise ValueError("logger must be a Logger or a callable")

        self._format = time_format

    def __enter__(self):
        self.logger(f"Starting {self._name}...")
        self.current_timer = self._name
        self._start_time = perf_counter()
        self.timers_stack.append(self._name)
        return self

    def __exit__(self, *exc_info: tuple):
        self._stop_time = perf_counter()
        elapsed = self._stop_time - self._start_time
        formatted = self._format.format(elapsed, name=self._name)
        self.timers["/".join(self.timers_stack)] = elapsed
        self.timers_stack.pop(-1)
        self.logger(formatted)

    @classmethod
    def recap(cls, logger: Logger | Callable | None = None) -> str:
        """Return a string summarizing all the registered timers."""
        cls.timers["Total"] = sum([t for n, t in cls.timers.items() if n != "Total"])
        ret = ", ".join([f"{name}: {t:.2f}s" for name, t in cls.timers.items()])
        if isinstance(logger, Logger):
            logger.info(ret)
        elif callable(logger):
            logger(ret)
        elif logger is not None:
            raise ValueError("logger must be a Logger or a callable")
        return ret

    @classmethod
    def reset(cls) -> None:
        """Reset all the registered timers."""
        cls.timers = {}

    @classmethod
    def get_timer(cls, name: str) -> float:
        """Return the elapsed time of a timer."""
        return cls.timers[name]
