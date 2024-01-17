import logging
import os
from typing import Optional

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PlanetDataset


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
        self._test_transforms = get_transforms(width=config.width, height=config.height, augmentations=False)
        self._image_folder = os.path.join(config.dataset_path, 'images')

        self.train_dataset: Dataset
        self.valid_dataset: Dataset
        self.test_dataset: Dataset
    
    def prepare_data(self) -> None:
        """Checks data existance and log datasets info. 
        It will be called once before any worker start
        """
        # Check data quality
        train_ds_path = os.path.join(self._dataset_path, 'df_train.csv')
        valid_ds_path = os.path.join(self._dataset_path, 'df_valid.csv')
        test_ds_path = os.path.join(self._dataset_path, 'df_test.csv')

        # Ensure data exists
        if any(map(lambda x: not os.path.exists(x), [train_ds_path, valid_ds_path, test_ds_path])):
            raise RuntimeError('Not all dataframe files exist! Run "preprocess_data" first')

        ds_path = os.path.join(self._dataset_path, 'labels.csv')
        df = pd.read_csv(ds_path, index_col=0)
        logging.info(f'Original dataset: {ds_path} {len(df)}')
        df = df.drop_duplicates()
        logging.info(f'Without duplicates len: {len(df)}')

        train_df = pd.read_csv(train_ds_path)
        valid_df = pd.read_csv(valid_ds_path)
        test_df = pd.read_csv(test_ds_path)

        logging.info(f'Train dataset len: {len(train_df)}')
        logging.info(f'Valid dataset len: {len(valid_df)}')
        logging.info(f'Valid dataset len: {len(test_df)}')

        logging.info(f'Train dataset sample: {train_df.iloc[:5].image_name.values}')
        logging.info(f'Valid dataset sample: {valid_df.iloc[:5].image_name.values}')
        logging.info(f'Test dataset sample: {test_df.iloc[:5].image_name.values}')

        train_df_stat = train_df.drop(columns='image_name').sum(axis=0)
        valid_df_stat = valid_df.drop(columns='image_name').sum(axis=0)
        test_df_stat = test_df.drop(columns='image_name').sum(axis=0)

        logging.info(f'Train statistics: \n{(train_df_stat / train_df_stat.sum() * 100).to_string()}')
        logging.info(f'Valid statistics: \n{(valid_df_stat / valid_df_stat.sum() * 100).to_string()}')
        logging.info(f'Test statistics: \n{(test_df_stat / test_df_stat.sum() * 100).to_string()}')

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

            self.valid_dataset = PlanetDataset(
                labels_df=valid_df,
                images_folder=self._image_folder,
                transforms=self._valid_transforms,
            )
        elif stage == 'test':
            test_df = read_df(self._dataset_path, 'test')

            self.test_dataset = PlanetDataset(
                labels_df=test_df,
                images_folder=self._image_folder,
                transforms=self._test_transforms,
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
    
    def test_dataloader(self) -> DataLoader:
        """Returns dataloader with validation data

        Returns:
            DataLoader: loader with validation data
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """Reads dataframe after splitting from the specified data path

    Args:
        data_path (str): path to the csv file
        mode (str): string value for getting train, val or test datasets

    Returns:
        pd.DataFrame: dataframe loaded from disk
    """
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
