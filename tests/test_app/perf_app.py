#!/usr/bin/env python3
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import time

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    start = time.time()
    # keep the CPU busy for 5 seconds
    while time.time() - start < 3:
        a = 2347839287429375
        b = 49875032845
        _ = a * b + 8394750932.0


if __name__ == "__main__":
    my_app()
