"""Test for callbacks.py module."""

import os
import shutil
from pathlib import Path
from textwrap import dedent
import contextlib
import pytest
from datetime import datetime
import git
from hydra.test_utils.test_utils import (
    assert_regex_match,
    _chdir_to_dir_containing,
    run_python_script,
)
import hydra_callbacks.callbacks as callbacks  # noqa: F401
import pandas as pd
import numpy as np

_chdir_to_dir_containing("pyproject.toml")


@contextlib.contextmanager  # type: ignore
def chdirto(new_dir) -> None:  # type: ignore
    """Very simple context manager to change directory temporarly.

    https://stackoverflow.com/a/75049063
    """
    d = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(d)


HYDRA_LAUNCH_LOG = "[HYDRA] Launching 1 jobs locally\n[HYDRA] 	#0 : \n"


@pytest.mark.parametrize(
    "config,args,expected",
    [
        pytest.param("config.yaml", [], "[JOB] foo: bar", id="no_callback"),
        pytest.param(
            "git_callback.yaml",
            ["hydra.callbacks.git_infos.clean=false"],
            dedent(
                """\
                [HYDRA] Git sha: {sha}, dirty: {dirty}
                [JOB] foo: bar
                git:
                  sha: {sha}
                  is_dirty: {dirty_lower}
                """.format(
                    sha=git.Repo().head.object.hexsha,
                    dirty=str(git.Repo().is_dirty()),
                    dirty_lower=str(git.Repo().is_dirty()).lower(),
                )
            ),
            id="git_callback",
        ),
        pytest.param(
            "runtime_perf.yaml",
            ["hydra.callbacks.runtime_perf.enabled=true"],
            dedent(
                """\
                [JOB] foo: bar

                \\[JOB\\] Total runtime: [0-9]+\\.[0-9]{2} seconds
                """
            ),
            id="runtime_perf",
        ),
        pytest.param(
            "runtime_perf.yaml",
            ["hydra.callbacks.runtime_perf.enabled=false"],
            dedent(
                """\
                [JOB] foo: bar
                """
            ),
            id="runtime_perf",
        ),
    ],
)
def test_app_with_callback_logs(
    tmpdir: Path,
    args: list[str],
    config: str,
    expected: str,
) -> None:
    """Test for app with callback which outputs messages."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=" + config,
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    cmd.extend(args)
    result, _err = run_python_script(cmd)
    assert _err == ""
    assert_regex_match(expected, result)


# same test as above but with multirun enabled.
@pytest.mark.parametrize(
    "config,args,expected",
    [
        pytest.param(
            "config.yaml",
            [],
            HYDRA_LAUNCH_LOG
            + dedent(
                """
                [JOB] foo: bar
                """
            ),
            id="no_callback",
        ),
        pytest.param(
            "git_callback.yaml",
            ["hydra.callbacks.git_infos.clean=false"],
            dedent(
                """\
                [HYDRA] Git sha: {sha}, dirty: {dirty}
                """.format(
                    sha=git.Repo().head.object.hexsha,
                    dirty=str(git.Repo().is_dirty()),
                )
            )
            + HYDRA_LAUNCH_LOG
            + "[JOB] foo: bar",  # git info in multirun are in the multirun.yaml file.
            id="git_callback",
        ),
        pytest.param(
            "runtime_perf.yaml",
            ["hydra.callbacks.runtime_perf.enabled=true"],
            HYDRA_LAUNCH_LOG
            + dedent(
                """\
                [JOB] foo: bar

                \\[HYDRA\\] Total runtime: [0-9]+\\.[0-9]{2} seconds
                """
            ),
            id="runtime_perf",
        ),
        pytest.param(
            "runtime_perf.yaml",
            ["hydra.callbacks.runtime_perf.enabled=false"],
            HYDRA_LAUNCH_LOG
            + dedent(
                """\
                [JOB] foo: bar
                """
            ),
            id="runtime_perf",
        ),
    ],
)
def test_multirun_app_with_callback_logs(
    tmpdir: Path,
    args: list[str],
    config: str,
    expected: str,
) -> None:
    """Test for app with callback which outputs messages."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=" + config,
        "--multirun",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    cmd.extend(args)
    result, _err = run_python_script(cmd)
    assert_regex_match(expected, result)


@pytest.mark.parametrize("multirun", [True, False])
def test_latest_callback(tmpdir: Path, multirun: bool) -> None:
    """Test for latest callback."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=latest_callback",
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/run_one",
        "hydra.sweep.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/run_one",
        "hydra.callbacks.latest_callback.run_base_dir=" + str(tmpdir),
        "hydra.callbacks.latest_callback.multirun_base_dir=" + str(tmpdir),
        "hydra.job.chdir=False",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    cmd.insert(2, "--multirun") if multirun else None
    result, _err = run_python_script(cmd)
    assert _err == ""
    assert_regex_match(
        (HYDRA_LAUNCH_LOG if multirun else "")
        + dedent(
            r"""
            [JOB] foo: bar

            [{logger}] Latest run is at: {tmpdir}/{now}/run_one
            [{logger}] Latest run is also at: {tmpdir}/latest
            """.format(
                tmpdir=tmpdir,
                logger="HYDRA" if multirun else "JOB",
                now=datetime.now().strftime("%Y-%m-%d"),
            )
        ),
        result,
    )
    first_run_dir = (tmpdir / "latest").readlink()
    cmd[2] = "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/run_two"
    cmd[3] = "hydra.sweep.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/run_two"
    result2, _err2 = run_python_script(cmd)
    next_run_dir = (tmpdir / "latest").readlink()
    assert first_run_dir != next_run_dir
    # Do it again, the symlink should be updated


def test_multirun_gatherer(tmpdir: Path) -> None:
    """Test for multirun gatherer."""

    cmd = [
        "tests/test_app/gather_app.py",
        "--config-name=gather_app_conf.yaml",
        "--multirun",
        "+a=1,2,3",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    result, _err = run_python_script(cmd)
    assert _err == ""
    assert_regex_match(
        dedent(
            f"""\
            [HYDRA] Launching 3 jobs locally
            [HYDRA] 	#0 : +a=1
            [JOB] foo: bar
            a: 1

            [HYDRA] 	#1 : +a=2
            [JOB] foo: bar
            a: 2

            [HYDRA] 	#2 : +a=3
            [JOB] foo: bar
            a: 3

            [HYDRA] Gathered results in {tmpdir}/agg_results.csv
            """
        ),
        result,
    )
    with open(tmpdir / "agg_results.csv") as f:
        assert len(f.readlines()) == 4


def test_dirty_git_repo_error(tmpdir: Path) -> None:
    """Test for dirty git repo error."""

    cmd = [
        "dummy_app.py",
        "--config-name=git_callback.yaml",
        "hydra.callbacks.git_infos.clean=true",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    app_dir = tmpdir / "test_app"
    shutil.copytree("tests/test_app/", app_dir)
    # work in fake temporary repo.
    with chdirto(app_dir):
        testapp = git.Repo.init()
        testapp.git.add(".")
        testapp.index.commit("Initial commit")
        sha = testapp.head.object.hexsha
        # make the repo dirty
        with open("dummy.txt", "w") as f:
            f.write("Dummy has changed.")
        # run the test app.
        result, _err = run_python_script(cmd, raise_exception=False)
    assert _err == ""
    assert_regex_match(
        dedent(
            f"""\
                [HYDRA] Git sha: {sha}, dirty: True
                [HYDRA] Repo is dirty, aborting
                """
        ),
        result,
    )


@pytest.mark.parametrize("multirun", [True, False])
def test_resource_monitor(tmpdir: Path, multirun) -> None:
    """Test for resource monitor callback."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=resource_monitor.yaml",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    if multirun:
        cmd.insert(2, "--multirun")
    result, _err = run_python_script(cmd)
    assert_regex_match(
        (HYDRA_LAUNCH_LOG if multirun else "")
        + dedent(
            """\
        [JOB] foo: bar

        [{logger}] Writing monitoring data to {tmpdir}/resource_monitoring.csv
        """.format(
                tmpdir=tmpdir, logger="HYDRA" if multirun else "JOB"
            )
        ),
        result,
    )


def test_resource_monitor_disabled(tmpdir):
    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=resource_monitor.yaml",
        "hydra.callbacks.resource_monitor.enabled=False",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    result, _err = run_python_script(cmd)

    assert_regex_match(result, "[JOB] foo: bar")


def test_resource_monitor_raises(tmpdir):
    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=resource_monitor.yaml",
        "hydra.callbacks.resource_monitor.sample_interval=0.1",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    result, _err = run_python_script(cmd, raise_exception=False)

    assert "RuntimeError: Sampling interval (0.10s) cannot be lower than 0.2s" in _err


def test_resource_monitor_results(tmpdir: Path) -> None:
    """Test for resource monitor callback."""
    sampling_time = 0.3  # seconds
    cmd = [
        "tests/test_app/perf_app.py",
        "--config-name=resource_monitor.yaml",
        "hydra.callbacks.resource_monitor.sample_interval=" + str(sampling_time),
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    result, _err = run_python_script(cmd, raise_exception=False)

    df = pd.read_csv(tmpdir / "resource_monitoring.csv", index_col=0)
    # check that the sampling interval is respected
    assert len(df) >= 3 / 0.3
    np.testing.assert_allclose(df["time"].diff().mean(), sampling_time, rtol=0.05)
    # check that we got some activity on the cpu.
    assert df["cpus"].mean() > 0


@pytest.mark.parametrize("multirun", [True, False])
def test_register_callbacks(tmpdir: Path, multirun: bool) -> None:
    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=register_callback",
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d_%H-%M-%S}/",
        "hydra.sweep.dir=" + str(tmpdir) + "/sweep/${now:%Y-%m-%d_%H-%M-%S}/",
        "++hydra.callbacks.register_callback.run_base_dir=" + str(tmpdir),
        "++hydra.callbacks.register_callback.multirun_base_dir="
        + str(tmpdir)
        + "/sweep",
        "hydra.job.chdir=false",
    ]

    cmd.insert(2, "--multirun") if multirun else None
    result, _err = run_python_script(cmd)
    assert _err == ""
    cmd.insert(-1, "++foo=barbar")
    result, _err = run_python_script(cmd)
    assert _err == ""
    # FIXME Also parse the CSV !


@pytest.mark.parametrize("multirun", [True, False])
def test_env_callback(tmpdir: Path, multirun: bool) -> None:
    cmd = [
        "tests/test_app/setenv_app.py",
        "--config-name=setenv_callback",
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d_%H-%M-%S}/",
        "hydra.sweep.dir=" + str(tmpdir) + "/sweep/${now:%Y-%m-%d_%H-%M-%S}/",
        "hydra.job.chdir=false",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    cmd.insert(2, "--multirun") if multirun else None
    result, _err = run_python_script(cmd)

    assert_regex_match(
        (HYDRA_LAUNCH_LOG if multirun else "")
        + dedent(
            """\
            [JOB] foo: bar

            [JOB] MY_ENV_VAR_VALUE
            [JOB] bar
            """
        ),
        result,
    )

    assert _err == ""


@pytest.mark.parametrize("multirun", [True, False])
def test_exec_shell(tmpdir: Path, multirun: bool) -> None:
    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=shell_callback",
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d_%H-%M-%S}/",
        "hydra.sweep.dir=" + str(tmpdir) + "/sweep/${now:%Y-%m-%d_%H-%M-%S}/",
        "hydra.job.chdir=false",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    cmd.insert(2, "--multirun") if multirun else None
    result, _err = run_python_script(cmd)

    assert_regex_match(
        (HYDRA_LAUNCH_LOG if multirun else "")
        + dedent(
            """\
            [JOB] foo: bar
            """
        )
        + ("multirun" if multirun else "single run"),
        result,
    )
    assert _err == ""
