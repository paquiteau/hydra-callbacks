"""Callback mechanism for hydra  jobs."""
import os
import errno
import psutil
import threading
import json
import glob

import logging
from hydra.experimental.callback import Callback
from hydra.core.utils import JobReturn
from hydra.utils import to_absolute_path
from hydra.types import TaskFunction
from omegaconf import DictConfig, open_dict

import pandas as pd
import numpy as np
import time


_MB = 1024.0**2


class AnyRunCallback(Callback):
    """Abstract Callback that execute on any run."""

    def on_run_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        self._on_anyrun_start(config, **kwargs)

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        self._on_anyrun_start(config, **kwargs)

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        pass

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        self._on_anyrun_end(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        self._on_anyrun_end(config, **kwargs)

    def _on_anyrun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        pass


class RuntimePerformance(AnyRunCallback):
    """Callback that log total runtime infos.

    Parameters
    ----------
    enabled : bool
        if True, will log the total runtime.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            self._on_anyrun_start = lambda *args, **kwargs: None
            self._on_anyrun_end = lambda *args, **kwargs: None

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        self.start_time = time.perf_counter()

    def _on_anyrun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after any run."""
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        logging.getLogger("hydra").info(f"Total runtime: {duration:.2f} seconds")


class GitInfo(AnyRunCallback):
    """
    Callback that check git infos and log them.

    Parameters
    ----------
    clean
        if True, will fail if the repo is not clean
    """

    def __init__(self, clean: bool = False):
        self.clean = clean

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        import git

        log = logging.getLogger("hydra")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        is_dirty = repo.is_dirty()
        log.warning(f"Git sha: {sha}, dirty: {is_dirty}")

        if is_dirty and self.clean:
            log.error("Repo is dirty, aborting")  # pragma: no cover
            # sys.exit(1) raises an error, that is catched by hydra.
            # os._exit exits directly by stopping the process.
            os._exit(1)  # pragma: no cover

        # Add git info to config
        with open_dict(config):  # disable hydra's config protection
            config.git = {"sha": sha, "is_dirty": is_dirty}


class MultiRunGatherer(Callback):
    """Define a callback to gather job results from json files after a multirun.

    Parameters
    ----------
    result_file: str
        name of the file to gathers from all the jobs.
    """

    def __init__(self, result_file: str = "results.json"):
        self.result_file = result_file

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        save_dir = config.hydra.sweep.dir
        os.chdir(save_dir)
        results = []
        for filename in glob.glob(f"*/{self.result_file}"):
            with open(filename) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results.extend(loaded)
                else:
                    results.append(loaded)
        df = pd.DataFrame(results)
        df.to_csv("agg_results.csv")


class LatestRunLink(Callback):
    """Callback that create a symlink to the latest run in the base output dir.

    Parameters
    ----------
    run_base_dir: str
        name of the basedir
    multirun_base_dir: str
        name of the basedir for multirun
    """

    def __init__(
        self, run_base_dir: str = "outputs", multirun_base_dir: str = "multirun"
    ):
        self.run_base_dir = to_absolute_path(run_base_dir)
        self.multirun_base_dir = to_absolute_path(multirun_base_dir)

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a single run."""
        self._on_anyrun_end(config.hydra.run.dir, self.run_base_dir)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a multi run."""
        self._on_anyrun_end(config.hydra.sweep.dir, self.multirun_base_dir)

    def _on_anyrun_end(self, run_dir: str, base_dir: str) -> None:
        latest_dir_path = os.path.join(base_dir, "latest")
        self._force_symlink(
            to_absolute_path(run_dir),
            to_absolute_path(latest_dir_path),
        )
        logging.getLogger("hydra").info(f"Latest run is at: {latest_dir_path}")

    def _force_symlink(self, src: str, dest: str) -> None:
        """Create a symlink from src to test, overwriting dest if necessary."""
        try:
            os.symlink(src, dest)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(dest)
                os.symlink(src, dest)
            else:
                raise e  # pragma: no cover


class ResourceMonitorThread(threading.Thread):
    """
    A ``Thread`` to monitor a specific PID with a certain frequency to a file.

    Adapted from: https://github.com/nipy/nipype/blob/master/nipype/utils/profiler.py

    Parameters
    ----------
    pid: int
        The process ID to monitor
    sample_period: float
        The time interval at wat which to sample the process, in seconds.
    fname: str
    """

    def __init__(self, pid: int, sample_period: int | float = 1, base_name: str = None):
        # Make sure psutil is imported
        import psutil

        if sample_period < 0.2:
            raise RuntimeError(
                "Frequency (%0.2fs) cannot be lower than 0.2s" % sample_period
            )

        fname = f"p{pid}_t{time()}_f{sample_period}"
        fname = ".{base_name}_{fname}" if base_name else f".{fname}"

        self._fname = os.path.abspath(fname)
        self._logfile = open(self._fname, "w")
        self._sampletime = sample_period

        # Leave process initialized and make first sample
        self._process = psutil.Process(pid)
        self._sample(cpu_interval=0.2)

        # Start thread
        threading.Thread.__init__(self)
        self._event = threading.Event()

    @property
    def fname(self) -> str:
        """Get the internal filename."""
        return self._fname

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
        except psutil.NoSuchProcess:
            pass

        # Iterate through child processes and get number of their threads
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

        for child in children:
            try:
                with child.oneshot():
                    cpu += child.cpu_percent()
                    mem_info = child.memory_info()
                    rss += mem_info.rss
                    vms += mem_info.vms
            except psutil.NoSuchProcess:
                pass

        print(f"{time()}, {cpu}, {rss / _MB}, {vms / _MB}", file=self._logfile)
        self._logfile.flush()

    def run(self) -> None:
        """
        Core monitoring function.

        Called by the ``start()`` method of threading.Thread.
        """
        start_time = time()
        wait_til = start_time
        while not self._event.is_set():
            self._sample()
            wait_til += self._sampletime
            self._event.wait(max(0, wait_til - time()))

    def stop(self) -> dict[str, float | None]:
        """Stop monitoring."""
        if not self._event.is_set():
            self._event.set()
            self.join()
            self._sample()
            self._logfile.flush()
            self._logfile.close()

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


class RessourceMonitor(AnyRunCallback):
    """Callback that samples the cpu and memory usage during job execution.

    The collected  data (cpu percent, memory usage) is written to a csv file.

    TODO: Add GPU support.

    Parameters
    ----------
    enabled : bool
        if True, will log the total runtime.
    sample_interval : float or int
        The time interval at wat which to sample the process, in seconds.
    monitoring_file : str
        The file to write the monitoring data to.
    """

    def __init__(
        self,
        enabled: bool = True,
        sample_interval: float = 1,
        monitoring_file: str = "resource_monitoring.csv",
    ):
        self.enabled = enabled
        if not self.enabled:
            self.on_job_start = lambda *args, **kwargs: None
            self.on_job_end = lambda *args, **kwargs: None
            return

        self.sample_interval = sample_interval
        self.monitoring_file = monitoring_file

    def on_run_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a single run."""
        self._on_anyrun_start(config.hydra.run.dir)

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a multi run."""
        self._on_anyrun_start(config.hydra.sweep.dir)

    def _on_anyrun_start(self, run_dir: str) -> None:
        """Configure the path for the monitoring file."""
        self.monitoring_file = os.path.join(
            to_absolute_path(run_dir), self.monitoring_file
        )

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: None
    ) -> None:
        """Execute before a single job."""
        job_full_id = f"{config.hydra.job.name}_{config.hydra.job.id}"
        self._monitor[job_full_id] = ResourceMonitorThread(
            os.getpid(),
            sample_period=self.sample_interval,
            base_name=job_full_id,
        )
        self._monitor[job_full_id].start()

    def on_job_end(
        self,
        config: DictConfig,
        job_return: JobReturn,
        **kwargs: None,
    ) -> None:
        """Execute after a single job."""
        sampled_data = self._monitor[
            f"{config.hydra.job.name}_{config.hydra.job.id}"
        ].stop()

        del self._monitor[f"{config.hydra.job.name}_{config.hydra.job.id}"]
        sampled_data["prof_dict"]["job_name"] = config.hydra.job.name
        sampled_data["prof_dict"]["job_id"] = config.hydra.job.id

        df = pd.DataFrame(sampled_data["prof_dict"])

        df.to_csv(
            self.monitoring_file,
            mode="a",
            header=not os.path.exists(self.monitoring_file),
        )
