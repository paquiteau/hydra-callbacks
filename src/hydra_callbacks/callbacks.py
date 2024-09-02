"""Callback mechanism for hydra  jobs."""

from __future__ import annotations
from typing import Callable, Any
import errno
import glob
import json
import logging
import os
from pathlib import Path
import time

import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from hydra.types import TaskFunction
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict, OmegaConf
import omegaconf

from .monitor import ResourceMonitorService

callback_logger = logging.getLogger("hydra.callbacks")


def dummy_run(config: DictConfig, **kwargs: None) -> None:
    """Do nothing."""
    pass


class AnyRunCallback(Callback):
    """Abstract Callback that execute on any run."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        callback_logger.debug("Init %s", self.__class__.__name__)

        self.enabled = enabled
        if not self.enabled:
            # don't do anything if not enabled
            self._on_anyrun_start = dummy_run  # type: ignore
            self._on_anyrun_end = dummy_run  # type: ignore

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

    def __init__(
        self,
        result_file: str = "results.json",
        aggregator: Callable[[list[str]], os.PathLike] | None = None,
    ):
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

    def _default_aggregator(self, files: list[str]) -> os.PathLike:
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
        callback_logger.info(f"Latest run is at: {run_dir}")
        callback_logger.info(f"Latest run is also at: {latest_dir_path}")

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
    gpu_monit: bool , default False
        Also monitor gpu data.
    """

    _monitor: dict[tuple[str, str], Any]

    def __init__(
        self,
        enabled: bool = True,
        sample_interval: float = 1,
        monitoring_file: str = "resource_monitoring.csv",
        gpu_monit: bool = False,
    ):
        super().__init__(enabled)
        self.gpu_monit = gpu_monit

        self.sample_interval = sample_interval
        self.monitoring_file = monitoring_file

    def on_run_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a single run."""
        self._set_monit_file(config.hydra.run.dir)

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a multi run."""
        self._set_monit_file(config.hydra.sweep.dir)

    def _set_monit_file(self, run_dir: str) -> None:
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
            gpu_monit=self.gpu_monit,
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
        self._monitor[job_full_id].stop()
        sampled_data = self._monitor[job_full_id].get_values()

        del self._monitor[job_full_id]
        if sampled_data:
            df = pd.DataFrame(sampled_data)
            df["job_name"] = job_full_id[0]
            df["job_id"] = job_full_id[1]
            df.to_csv(
                self.monitoring_file,
                mode="a",
                header=not os.path.exists(self.monitoring_file),
            )
            max_cpu = sampled_data["cpus"].max()
            max_mem = sampled_data["rss_GiB"].max()
            callback_logger.debug(
                f"{job_full_id[0]}(#{job_full_id[1]}): max cpu: {max_cpu:.2f}%,"
                f"max mem: {max_mem:.2f} GiB"
            )

    def _get_job_info(self) -> tuple[str, str]:
        """Get the job id."""
        hconf = HydraConfig.get()
        name = hconf.job.name
        try:
            id = hconf.job.id
        except omegaconf.errors.MissingMandatoryValue:
            id = "0"
        return name, id


class RegisterRunCallback(Callback):
    """Callback that register the run in a .csv file at the end of the run.

    Single and MultiRun are handled in different files. Note that this append one row
    per config. Only the config is being registered, not the possible output of the run.

    Parameters
    ----------
    enabled : bool
        if True, will register the run.

    register_file: str
        name of the file to register the run in.
    """

    def __init__(
        self,
        enabled: bool = True,
        register_file: str = "register.csv",
        run_base_dir: str = "outputs",
        multirun_base_dir: str = "multiruns",
    ):
        self.enabled = enabled
        self.register_file = register_file
        self.run_base_dir = to_absolute_path(run_base_dir)
        self.multirun_base_dir = to_absolute_path(multirun_base_dir)
        if not self.enabled:
            self.on_job_end = dummy_run
        else:
            self.on_job_end = self._on_job_end  # type: ignore

    def _on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: None
    ) -> None:
        """Execute after every job."""
        # The hydra part of the conf is not resolvable (frozen)
        # and we don't want to monitor it (if needed it would still be available in
        # ``.hydra`` folder)
        # Hence:
        conf_ = OmegaConf.to_container(config)
        conf_.pop("hydra")
        conf_ = DictConfig(conf_)
        OmegaConf.resolve(conf_)
        conf_ = OmegaConf.to_container(conf_)
        pandas_config = pd.json_normalize(conf_)
        if config.hydra.mode == "MULTIRUN":
            pandas_config["run-dir"] = config.hydra.sweep.dir
            base_dir = self.multirun_base_dir
        else:
            pandas_config["run-dir"] = config.hydra.run.dir
            base_dir = self.run_base_dir

        pandas_config["job_success"] = job_return.status == JobStatus.COMPLETED

        register_file = os.path.join(base_dir, self.register_file)
        try:
            df = pd.read_csv(register_file, index_col=None)
            df = pd.concat([df, pandas_config])
            df.to_csv(register_file, index=False)
        except FileNotFoundError:
            df = pd.DataFrame(pandas_config)
            df.to_csv(register_file, index=False)


class SetEnvironment(AnyRunCallback):
    """Set environment variables from the config.

    The variable are unset at the end of the run.
    """

    def __init__(self, enabled: bool = True, env: dict[str, str] | None = None):
        super().__init__(enabled=enabled)
        self.env = env or {}

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Set the environment variables."""
        for key, value in self.env.items():
            os.environ[key] = value

    def _on_any_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Unset the environment variables."""
        for key in self.env:
            os.environ.pop(key, None)


class ExecShellCommand(AnyRunCallback):
    """Execute a shell command at the end of the run."""

    def __init__(self, run_command: str = "", multirun_command: str = ""):
        self.run_command = run_command
        self.multirun_command = multirun_command

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a single run."""
        os.system(self.run_command)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute after a multi run."""
        os.system(self.multirun_command)
