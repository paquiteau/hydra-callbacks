import hydra
import logging
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
