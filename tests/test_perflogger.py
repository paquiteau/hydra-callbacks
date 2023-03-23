import pytest
import logging

from time import sleep
from hydra_callbacks import PerfLogger


@pytest.mark.parametrize(
    "func", [print, logging.getLogger("test_perflogger")], scope="function"
)
def test_perflogger(func):
    PerfLogger.reset()
    with PerfLogger(func, name="test1"):
        sleep(0.2)

    with PerfLogger(func, name="test2"):
        sleep(0.3)

    recap_str = PerfLogger.recap()

    assert recap_str == "test1: 0.20s, test2: 0.30s, Total: 0.50s"


def test_perflogger_raises():
    with pytest.raises(ValueError):
        with PerfLogger("Not Callable", name="test1"):
            sleep(1)
