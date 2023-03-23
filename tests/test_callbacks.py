"""Test for callbacks.py module."""

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


@pytest.mark.parametrize("multirun", [True, False])
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
    multirun: bool,
    config: str,
    expected: str,
) -> None:
    """Test for app with callback which outputs messages."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--multirun" if multirun else "",
        "--config-name=" + config,
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    cmd.extend(args)
    result, _err = run_python_script(cmd)
    assert_regex_match(expected, result)


def test_latest_callback(tmpdir: Path) -> None:
    """Test for latest callback."""

    cmd = [
        "tests/test_app/dummy_app.py",
        "--config-name=latest_callback",
        "hydra.run.dir=" + str(tmpdir) + "/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}",
        "hydra.callbacks.latest_callback.run_base_dir=" + str(tmpdir),
        "hydra.job.chdir=False",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    result, _err = run_python_script(cmd)
    assert_regex_match(
        dedent(
            """\
            [JOB] foo: bar

            [JOB] Latest run is at: {tmpdir}/latest
            """.format(
                tmpdir=tmpdir
            )
        ),
        result,
    )
    first_run_dir = (tmpdir / "latest").readlink()
    result2, _err2 = run_python_script(cmd)
    next_run_dir = (tmpdir / "latest").readlink()
    assert first_run_dir != next_run_dir
    # Do it again, the symlink should be updated
