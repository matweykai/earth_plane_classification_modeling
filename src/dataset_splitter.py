import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import IterativeStratification


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset on train/valid sets and saves it

    Args:
        annotation (pd.DataFrame): dataframe with annotations
        train_fraction (float, optional): specifies ratio size of the train dataset. Defaults to 0.8.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and validation
    """
    y_columns = [
        'clear',
        'partly_cloudy',
        'cloudy',
        'haze',
        'agriculture',
        'cultivation',
        'bare_ground',
        'conventional_mine',
        'artisinal_mine',
        'primary',
        'blooming',
        'selective_logging',
        'blow_down',
        'slash_burn',
        'habitation',
        'water',
        'road',
    ]

    all_x = annotation.index.to_numpy(np.uint8)
    all_y = annotation[y_columns].to_numpy(dtype=np.uint8)

    train_indexes, other_indexes = _split(all_x, all_y, distribution=[1 - train_fraction, train_fraction])

    train_subset = annotation.iloc[train_indexes]
    other_subset = annotation.iloc[other_indexes]

    test_indexes, valid_indexes = _split(other_subset.index.to_numpy(np.uint8), other_subset[y_columns].to_numpy(dtype=np.uint8), distribution=[0.5, 0.5])
    valid_subset = other_subset.iloc[valid_indexes]
    test_subset = other_subset.iloc[test_indexes]

    logging.info("Stratifying dataset is completed.")

    return train_subset, valid_subset, test_subset


def _split(
    xs: np.ndarray,
    ys: np.ndarray,
    distribution: Union[None, List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits multilabel dataset with sklearn multilearn function

    Args:
        xs (np.array): X values
        ys (np.array): Y values
        distribution (Union[None, List[float]], optional): Disttibution parameter for IterativeStratification function. Defaults to None.

    Returns:
        Tuple[np.array, np.array]: indexes for the first and second datasets
    """
    stratifier = IterativeStratification(n_splits=2, sample_distribution_per_fold=distribution)
    first_indexes, second_indexes = next(stratifier.split(X=xs, y=ys))

    return first_indexes, second_indexes
