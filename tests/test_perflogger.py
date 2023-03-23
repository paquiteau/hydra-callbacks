import pytest
import logging

from hydra_callbacks import PerfLogger
from time import sleep


@pytest.mark.parametrize("func", [print, logging.getLogger("test_perflogger")])
def test_perflogger(func):

    with PerfLogger(func, name="test1"):
        sleep(1)

    with PerfLogger(func, name="test2"):
        sleep(1)

    recap_str = PerfLogger.recap()

    assert recap_str == "test1: 1.00s, test2: 1.00s, Total: 2.00s"


def test_perflogger_raises():
    with pytest.raises(ValueError):
        with PerfLogger("Not Callable", name="test1"):
            sleep(1)
