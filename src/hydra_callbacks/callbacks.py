"""Callback mechanism for hydra  jobs."""
import os
import errno
import json
import glob

import logging
from hydra.experimental.callback import Callback
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict

import pandas as pd
import time


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
