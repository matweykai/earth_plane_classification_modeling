import os
import logging
import argparse
import pandas as pd

from src.config import Config
from src.dataset_splitter import stratify_shuffle_split_subsets


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    """Splits dataset into train and valid subsets and saves them in the same directory.
    Specified directory should contain 'labels.csv' file

    Args:
        data_path (str): path to the directory with images folder and labels.csv file
        train_fraction (float, optional): ratio of training samples in the final dataset. Defaults to 0.8.
    """
    ds_path = os.path.join(data_path, 'labels.csv')

    df = pd.read_csv(ds_path, index_col=0)
    df = df.drop_duplicates()

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, train_fraction=train_fraction)

    logging.info('Datasets splitted')

    train_ds_path = os.path.join(data_path, 'df_train.csv')
    valid_ds_path = os.path.join(data_path, 'df_valid.csv')
    test_ds_path = os.path.join(data_path, 'df_test.csv')

    train_df.to_csv(train_ds_path, index=False)
    valid_df.to_csv(valid_ds_path, index=False)
    test_df.to_csv(test_ds_path, index=False)

    logging.info('Datasets saved')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Get config from provided file
    parser = argparse.ArgumentParser()

    parser.add_argument('config_path', help='path to the train config file')
    arguments = parser.parse_args()

    config = Config.from_yaml(arguments.config_path).data_config

    split_and_save_datasets(config.dataset_path, config.train_fraq)
