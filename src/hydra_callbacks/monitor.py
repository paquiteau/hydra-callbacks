"""Resource Monitor Utilities."""
import logging
import multiprocessing
import os
import subprocess
import time
from typing import Callable, Mapping

import numpy as np
import psutil
from numpy.typing import ArrayLike

_MB = 1024.0**2


callback_logger = logging.getLogger("hydra.callbacks")


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
    interval: float or int
        The sampling interval in seconds, default 1s.
    base_name: str
        Base name for the monitoring trace file.
    gpu_monit: bool
        If True, also monitor gpu usage and memory, default False.
    """

    def __init__(
        self,
        pid: int,
        interval: int | float = 1,
        base_name: str = "",
        gpu_monit: bool = False,
        gpu_devices: list[int] | None = None,
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
        if gpu_monit:
            try:
                subprocess.check_call(
                    ["nvidia-smi", "-L"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:  # pragma: no cover
                callback_logger.warning("nvidia-smi failed, gpu profiling is disabled.")
                gpu_monit = False

        self.gpu_monit = gpu_monit
        if gpu_monit and gpu_devices is None:
            n_gpu = len(subprocess.check_output(["nvidia-smi", "-L"]).splitlines())
            self.gpu_devices = list(range(n_gpu))

        # Leave process initialized and make first sample
        self._process = psutil.Process(pid)

    def _sample(self, cpu_interval: float | None = None) -> None:
        cpu = 0.0
        rss = 0.0
        vms = 0.0
        pids = [self._process.pid]
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
                pids.append(child.pid)
                with child.oneshot():
                    cpu += child.cpu_percent()
                    mem_info = child.memory_info()
                    rss += mem_info.rss
                    vms += mem_info.vms
            except psutil.NoSuchProcess:  # pragma: no cover
                pass

        log_string = f"{time.time()}, {cpu}, {rss / _MB}, {vms / _MB}"
        if self.gpu_monit:
            gpu_mems, gpu_usages = self._gpu_sample(pids)
            for mem, usage in zip(gpu_mems, gpu_usages):
                log_string += f", {mem}, {usage}"
        print(log_string, file=self._logfile)
        self._logfile.flush()

    def _gpu_sample(self, pids: list[int]) -> tuple[list[int], list[int]]:
        """Sample the GPU usage."""
        mem = [0] * len(self.gpu_devices)
        usage = [0] * len(self.gpu_devices)
        pmon_mem = subprocess.check_output(["nvidia-smi", "pmon", "-c=1", "-s=m"])
        pmon_usage = subprocess.check_output(["nvidia-smi", "pmon", "-c=1", "-s=u"])
        # get Memory Frame Buffer Size
        for line in pmon_mem.splitlines()[2:]:
            sample = list(filter(None, line.split(b"  ")))
            pid = int(sample[1])
            if pid in pids:
                device = int(sample[0])
                mem[device] += int(sample[3])

        # get SM usage
        for line in pmon_usage.splitlines()[2:]:
            sample = list(filter(None, line.split(b"  ")))
            pid = int(sample[1])
            if pid in pids:
                device = int(sample[0])
                usage[device] += int(sample[4])

        return mem, usage

    def __del__(self) -> None:
        """Close the log file."""
        self._logfile.close()

    def start(self) -> None:
        """Start monitoring."""
        self._timer = ProcessTimer(self._interval, self._sample)
        self._sample(cpu_interval=0.2)
        self._timer.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._timer.cancel()
        del self._timer
        # Read .prof file in and set runtime values
        vals = np.loadtxt(self._fname, delimiter=",")
        if not vals.size:
            return None

        vals = np.atleast_2d(vals)
        valdict = {
            "time": vals[:, 0],
            "cpus": vals[:, 1],
            "rss_GiB": vals[:, 2] / 1024,
            "vms_GiB": vals[:, 3] / 1024,
        }
        if self.gpu_monit:
            for i in self.gpu_devices:
                valdict[f"gpu{i}_mem_GiB"] = vals[:, 4 + 2 * i] / 1024
                valdict[f"gpu{i}_usage"] = vals[:, 5 + 2 * i]
        self._valdict = valdict

    def get_values(self) -> Mapping[str, ArrayLike]:
        """Return the values collected by the monitor."""
        if not hasattr(self, "_valdict"):
            raise RuntimeError("You must call stop() before get_values().")
        return self._valdict

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info: tuple):
        self.stop()
