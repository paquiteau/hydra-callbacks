import hydra
import logging
import json
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    conf_dict = OmegaConf.to_container(cfg)
    if cfg.a == 1:
        # simulate a list of results.
        # this is usefull if  Each job run a batch of experiments.
        conf_dict = [conf_dict]
    with open("results.json", "w") as f:
        json.dump(conf_dict, f)


if __name__ == "__main__":
    my_app()
