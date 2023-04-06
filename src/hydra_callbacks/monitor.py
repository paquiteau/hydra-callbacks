"""Resource Monitor Utilities."""
import multiprocessing
import os
import time
from typing import Callable

import numpy as np
import psutil

_MB = 1024.0**2


class ProcessTimer(multiprocessing.Process):
    """Timer behavior using Process instead of Thread.

    Using a process make sure that we won't be block by the GIL.

    Adapted from: https://stackoverflow.com/a/25297758
    """

    def __init__(
        self,
        sample_interval: int | float,
        function: Callable,
        *args: None,
        **kwargs: None,
    ):
        super().__init__()
        self.interval = sample_interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = multiprocessing.Event()

    def cancel(self) -> None:
        """Stop the timer if it hasn't finished yet."""
        self.finished.set()

    def run(self) -> None:
        """Run the timer."""
        while not self.finished.wait(self.interval):  # pragma: no cover
            self.function(*self.args, **self.kwargs)


class ResourceMonitorService:
    """
    A Service that monitor a PID with a certain frequency to a file.

    A threading or a process backend is available. The process backend is recommended
    for single core application that do not release the GIL. The threading backend is
    recommended for multi core / multi processes application that release the GIL

    Adapted from: https://github.com/nipy/nipype/blob/master/nipype/utils/profiler.py


    Parameters
    ----------
    pid: int
        The process ID to monitor
    sample_period: float
        The time interval at wat which to sample the process, in seconds.
    base_name: str
        The name of the job, used to determined the temporary file name.
    backend: str
        The backend to use. Can be either "threading" or "process"
    """

    def __init__(
        self,
        pid: int,
        interval: int | float = 1,
        base_name: str = None,
    ):
        # Make sure psutil is imported
        import psutil

        if interval < 0.2:
            raise RuntimeError(
                f"Sampling interval ({interval:0.2f}s) cannot be lower than 0.2s"
            )

        fname = f"p{pid}_t{time.time()}_f{interval}"
        fname = f".{base_name}_{fname}" if base_name else f".{fname}"

        self._fname = os.path.abspath(fname)
        self._logfile = open(self._fname, "w")
        self._interval = interval

        # Leave process initialized and make first sample
        self._process = psutil.Process(pid)
        self._sample(cpu_interval=0.2)

        # Start thread
        self._timer = ProcessTimer(self._interval, self._sample)

    def _sample(self, cpu_interval: float = None) -> None:
        cpu = 0.0
        rss = 0.0
        vms = 0.0
        try:
            with self._process.oneshot():
                cpu += self._process.cpu_percent(interval=cpu_interval)
                mem_info = self._process.memory_info()
                rss += mem_info.rss
                vms += mem_info.vms
        except psutil.NoSuchProcess:  # pragma: no cover
            pass

        # Iterate through child processes and get number of their threads
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess:  # pragma: no cover
            children = []

        for child in children:
            try:  # pragma: no cover
                with child.oneshot():
                    cpu += child.cpu_percent()
                    mem_info = child.memory_info()
                    rss += mem_info.rss
                    vms += mem_info.vms
            except psutil.NoSuchProcess:  # pragma: no cover
                pass

        print(f"{time.time()}, {cpu}, {rss / _MB}, {vms / _MB}", file=self._logfile)
        self._logfile.flush()

    def start(self) -> None:
        """Start monitoring."""
        self._sample(cpu_interval=0.2)
        self._timer.start()

    def stop(self) -> dict[str, float | None]:
        """Stop monitoring."""
        self._timer.cancel()
        del self._timer

        retval = {
            "mem_peak_gb": None,
            "cpu_percent": None,
        }

        # Read .prof file in and set runtime values
        vals = np.loadtxt(self._fname, delimiter=",")
        if vals.size:
            vals = np.atleast_2d(vals)
            retval["mem_peak_gb"] = vals[:, 2].max() / 1024
            retval["cpu_peak_percent"] = vals[:, 1].max()
            retval["prof_dict"] = {
                "time": vals[:, 0],
                "cpus": vals[:, 1],
                "rss_GiB": vals[:, 2] / 1024,
                "vms_GiB": vals[:, 3] / 1024,
            }
        return retval
