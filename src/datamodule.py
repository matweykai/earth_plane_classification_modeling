import logging
import os
from typing import Optional, Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PlanetDataset
from src.dataset_splitter import stratify_shuffle_split_subsets


class PlanetDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        """Inits Planet data module with specified config

        Args:
            config (DataConfig): config with data loading settings
        """
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.n_workers
        self._train_fraq = config.train_fraq
        self._dataset_path = config.dataset_path
        self._train_transforms = get_transforms(width=config.width, height=config.height)
        self._valid_transforms = get_transforms(width=config.width, height=config.height, augmentations=False)
        self._image_folder = os.path.join(config.data_path, 'images')

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        """Prepares data for training. It will be called once before any worker start
        """
        split_and_save_datasets(self._dataset_path, self._train_fraq)

    def setup(self, stage: Optional[str] = None):
        """Setups each worker environment for training and validation purposes

        Args:
            stage (Optional[str], optional): specifies stage of the model training (Can be 'fit' and 'test'). Defaults to None.
        """
        if stage == 'fit':
            train_df = read_df(self._dataset_path, 'train')

            self.train_dataset = PlanetDataset(
                labels_df=train_df,
                images_folder=self._image_folder,
                transforms=self._train_transforms,
            )

            valid_df = read_df(self._dataset_path, 'valid')

            self.train_dataset = PlanetDataset(
                labels_df=valid_df,
                images_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns dataloader with training data

        Returns:
            DataLoader: loader with training data
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,

        )

    def val_dataloader(self) -> DataLoader:
        """Returns dataloader with validation data

        Returns:
            DataLoader: loader with validation data
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,

        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    """Splits dataset into train and valid subsets and saves them in the same directory.
    Specified directory should contain 'labels.csv' file

    Args:
        data_path (str): path to the directory with images folder and labels.csv file
        train_fraction (float, optional): ratio of training samples in the final dataset. Defaults to 0.8.
    """
    ds_path = os.path.join(data_path, 'labels.csv')

    df = pd.read_csv(ds_path)
    logging.info(f'Original dataset: {ds_path} {len(df)}')
    df = df.drop_duplicates()
    logging.info(f'Without duplicates len: {len(df)}')

    train_df, valid_df = stratify_shuffle_split_subsets(df, train_fraction=train_fraction)
    logging.info(f'Train dataset len: {len(train_df)}')
    logging.info(f'Valid dataset len: {len(valid_df)}')

    train_ds_path = os.path.join(data_path, 'df_train.csv')
    valid_ds_path = os.path.join(data_path, 'df_valid.csv')

    logging.info(f'Train dataset_path: {train_ds_path}')
    logging.info(f'Valid dataset_path: {valid_ds_path}')

    train_df.to_csv(train_ds_path, index=False)
    valid_df.to_csv(valid_ds_path, index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """Reads dataframe after splitting from the specified data path

    Args:
        data_path (str): path to the csv file
        mode (str): string value for getting train or val datasets

    Returns:
        pd.DataFrame: dataframe loaded from disk
    """
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
