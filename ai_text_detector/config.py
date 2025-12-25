from hydra import compose, initialize
from omegaconf import DictConfig


def load_config(overrides=None) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg
