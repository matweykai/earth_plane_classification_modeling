from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Pydantic model class for storing data loading parameters"""
    dataset_path: str
    batch_size: int
    n_workers: int
    train_fraq: float
    width: int
    height: int


class LossConfig(BaseModel):
    """Pydantic model class for storing loss functions parameters"""
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class Config(BaseModel):
    """Class for storing training parameters"""
    data_config: DataConfig
    losses: List[LossConfig]
    n_epochs: int
    num_classes: int
    monitor_metric: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict


    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Loads config from yaml file

        Args:
            path (str): path to yaml config file

        Returns:
            Config: config object that stores all training settings
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
