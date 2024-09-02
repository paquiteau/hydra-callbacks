#!/usr/bin/env python3
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info(os.getenv("MY_ENV_VAR"))
    log.info(os.getenv("MY_ENV_VAR_RESOLVED"))


if __name__ == "__main__":
    my_app()
