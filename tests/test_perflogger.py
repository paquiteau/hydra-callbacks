import pytest
import logging
from numpy.testing import assert_allclose
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

    recap_str = PerfLogger.recap(func)
    recap_str2 = PerfLogger.recap()

    assert recap_str2 == recap_str == "test1: 0.20s, test2: 0.30s, Total: 0.50s"
    # sleep is not precise, so we use a large tolerance
    assert_allclose(PerfLogger.get_timer("test1"), 0.2, atol=1e-3, rtol=1e-2)
    assert_allclose(PerfLogger.get_timer("test2"), 0.3, atol=1e-3, rtol=1e-2)


def test_perflogger_raises():
    with pytest.raises(ValueError):
        with PerfLogger("Not Callable", name="test1"):
            sleep(1)

    with pytest.raises(ValueError):
        with PerfLogger(print, name="test2"):
            sleep(1)
        PerfLogger.recap("Not Callable")
