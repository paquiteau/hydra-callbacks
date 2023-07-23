"""Callback mechanism for hydra  jobs."""
from __future__ import annotations
from typing import Callable
import errno
import glob
import json
import logging
import os
from pathlib import Path
import time

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.types import TaskFunction
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict

from .monitor import ResourceMonitorService

callback_logger = logging.getLogger("hydra.callbacks")


class AnyRunCallback(Callback):
    """Abstract Callback that execute on any run."""

    def __init__(self, enabled: bool = True):
        callback_logger.debug("Init %s", self.__class__.__name__)

        self.enabled = enabled
        if not self.enabled:
            # don't do anything if not enabled
            self.on_job_start = lambda *args, **kwargs: None
            self.on_job_end = lambda *args, **kwargs: None
            self._on_anyrun_start = lambda *args, **kwargs: None
            self._on_anyrun_end = lambda *args, **kwargs: None

    def on_run_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        callback_logger.debug("run start callback %s", self.__class__.__name__)
        self._on_anyrun_start(config, **kwargs)

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        callback_logger.debug("(multi)run start callback %s", self.__class__.__name__)
        self._on_anyrun_start(config, **kwargs)

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        pass

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        callback_logger.debug("run end callback %s", self.__class__.__name__)
        self._on_anyrun_end(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        callback_logger.debug("(multi)run end callback %s", self.__class__.__name__)
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
        super().__init__(enabled)

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        self.start_time = time.perf_counter()

    def _on_anyrun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after any run."""
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        callback_logger.info(f"Total runtime: {duration:.2f} seconds")


class GitInfo(AnyRunCallback):
    """
    Callback that check git infos and log them.

    Parameters
    ----------
    clean
        if True, will fail if the repo is not clean
    """

    def __init__(self, clean: bool = False):
        super().__init__()
        self.clean = clean

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        import git

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        is_dirty = repo.is_dirty()
        callback_logger.warning(f"Git sha: {sha}, dirty: {is_dirty}")

        if is_dirty and self.clean:
            callback_logger.error("Repo is dirty, aborting")  # pragma: no cover
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
    aggregator: Callable
        function to aggregate the results. This function should take a list of
        filepath as input and process them. By default this assume that each
        result file is a json file, and will load them as a list of dict, and
        save them as a csv file.
    """

    def __init__(self, result_file: str = "results.json", aggregator: Callable = None):
        callback_logger.debug("Init %s", self.__class__.__name__)
        self.result_file = result_file

        self.aggregator = aggregator or self._default_aggregator

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run after all job have ended.

        Will write a DataFrame from all the results. at the run location.
        """
        save_dir = config.hydra.sweep.dir
        os.chdir(save_dir)
        self.aggregator(glob.glob(f"*/{self.result_file}"))

    def _default_aggregator(self, files: list[os.PathLike]) -> os.PathLike:
        """Aggregat the results as a dataframe and save it as csv."""
        results = []
        for filename in files:
            with open(filename) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    results.extend(loaded)
                else:
                    results.append(loaded)
        df = pd.DataFrame(results)
        df.to_csv("agg_results.csv")
        callback_logger.info(f"Gathered results in {Path.cwd() / 'agg_results.csv'}")
        return Path.cwd() / "agg_results.csv"


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
        callback_logger.debug("Init %s", self.__class__.__name__)
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
        callback_logger.info(f"Latest run is at: {latest_dir_path}")

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


class ResourceMonitor(AnyRunCallback):
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
        super().__init__(enabled)

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
        self._monitor = {}

    def _on_anyrun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Run on any run end."""
        callback_logger.info(f"Writing monitoring data to {self.monitoring_file}")

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: None
    ) -> None:
        """Execute before a single job."""
        job_full_id = self._get_job_info()
        self._monitor[job_full_id] = ResourceMonitorService(
            os.getpid(),
            interval=self.sample_interval,
            base_name=f"{job_full_id[0]},{job_full_id[1]}",
        )
        self._monitor[job_full_id].start()

    def on_job_end(
        self,
        config: DictConfig,
        job_return: JobReturn,
        **kwargs: None,
    ) -> None:
        """Execute after a single job."""
        job_full_id = self._get_job_info()
        sampled_data = self._monitor[job_full_id].stop()

        del self._monitor[job_full_id]
        sampled_data["prof_dict"]["job_name"] = job_full_id[0]
        sampled_data["prof_dict"]["job_id"] = job_full_id[1]

        df = pd.DataFrame(sampled_data["prof_dict"])
        df.to_csv(
            self.monitoring_file,
            mode="a",
            header=not os.path.exists(self.monitoring_file),
        )

    def _get_job_info(self) -> str:
        """Get the job id."""
        hconf = HydraConfig.get()
        name = hconf.job.name
        id = hconf.job.get("id", 0)
        return name, id
