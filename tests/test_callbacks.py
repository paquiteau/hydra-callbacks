"""Test for callbacks.py module."""

import os
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import git
from hydra.test_utils.test_utils import (
    assert_regex_match,
    _chdir_to_dir_containing,
    run_process,
    run_python_script,
)
from omegaconf import open_dict, read_write

_chdir_to_dir_containing("pyproject.toml")


@pytest.mark.parametrize(
    "config,expected",
    [
        pytest.param("config.yaml", "[JOB] foo: bar", id="no_callback"),
        pytest.param("git_callback.yaml", "[JOB] foo: bar", id="git_callback"),
    ],
)
def test_app_with_callback(
    tmpdir: Path,
    config: str,
    expected: str,
) -> None:
    """Test for app with callback."""

    cmd = [
        "tests/test_app/my_app.py",
        "--config-name=" + config,
        "hydra.run.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        "hydra.hydra_logging.formatters.simple.format='[HYDRA] %(message)s'",
        "hydra.job_logging.formatters.simple.format='[JOB] %(message)s'",
    ]
    result, _err = run_python_script(cmd, raise_exception=False)

    assert_regex_match(expected, result)
