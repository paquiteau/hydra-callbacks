"""A collection of Hydra callbacks for logging and performance analysis."""

from .callbacks import (
    AnyRunCallback,
    GitInfo,
    LatestRunLink,
    MultiRunGatherer,
    ResourceMonitor,
    RuntimePerformance,
    RegisterRun,
    SetEnvironment,
    ExecShellCommand,
)
from .logger import PerfLogger

__all__ = [
    "AnyRunCallback",
    "RuntimePerformance",
    "GitInfo",
    "MultiRunGatherer",
    "ResourceMonitor",
    "LatestRunLink",
    "PerfLogger",
    "RegisterRun",
    "SetEnvironment",
    "ExecShellCommand",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
