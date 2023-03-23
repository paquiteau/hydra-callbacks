"""Test for callbacks.py module."""
import os
import shutil
from pathlib import Path
from textwrap import dedent

import pytest
import git
from hydra.test_utils.test_utils import (
    assert_regex_match,
    _chdir_to_dir_containing,
    run_python_script,
)
import hydra_callbacks

_chdir_to_dir_containing("pyproject.toml")


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
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}",
        "hydra.sweep.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}",
        "hydra.callbacks.latest_callback.run_base_dir=" + str(tmpdir),
        "hydra.callbacks.latest_callback.multirun_base_dir=" + str(tmpdir),
        "hydra.job.chdir=False",
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

            [{logger}] Latest run is at: {tmpdir}/latest
            """.format(
                tmpdir=tmpdir, logger="HYDRA" if multirun else "JOB"
            )
        ),
        result,
    )
    first_run_dir = (tmpdir / "latest").readlink()
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

    assert_regex_match(
        result,
        dedent(
            """
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
            """
        ),
    )
    with open(tmpdir / "agg_results.csv") as f:
        assert f.read() == dedent(
            """\
            ,foo,a
            0,bar,3
            1,bar,2
            2,bar,1
            """
        )


def test_dirty_git_repo_error(tmpdir: Path) -> None:
    """Test for dirty git repo error."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=git_callback.yaml",
        "hydra.callbacks.git_infos.clean=true",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]

    dirty = git.Repo().is_dirty()

    if not dirty:
        shutil.copy2("tests/test_app/dummy.txt", "tests/test_app/dummy.bak")
        # create a dummy file to make the repo dirty
        with open("tests/test_app/dummy.txt", "w") as f:
            f.write("Dummy has changed.")

    result, _err = run_python_script(cmd, raise_exception=False)
    assert _err == ""
    assert_regex_match(
        result,
        dedent(
            """\
                [HYDRA] Git sha: {sha}, dirty: True
                [HYDRA] Repo is dirty, aborting
                """.format(
                sha=git.Repo().head.object.hexsha,
            )
        ),
    )

    if not dirty:
        shutil.copy2("tests/test_app/dummy.bak", "tests/test_app/dummy.txt")
