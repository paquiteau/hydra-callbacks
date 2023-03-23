import pytest

from hydra_callbacks import PerfLogger
from time import sleep


def test_perflogger():

    with PerfLogger(print, name="test1"):
        sleep(1)

    with PerfLogger(print, name="test2"):
        sleep(1)

    recap_str = PerfLogger.recap()

    assert recap_str == "test1: 1.00s, test2: 1.00s, Total: 2.00s"
