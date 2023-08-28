import pytest
import numpy as np
import pandas as pd

from src.dataset_splitter import stratify_shuffle_split_subsets


@pytest.mark.parametrize(
    ('ds_size', 'train_ratio', 'expected_train_size'),
    [
        (10, 0.8, 8),
        (100, 0.7, 70),
        (1000, 0.3, 300),
    ]
)
def test_train_ratio_split(ds_size: int, train_ratio: float, expected_train_size: int):
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

    file_names = [f'file_{ind}.jpg' for ind in range(ds_size)]
    np_dataset = np.hstack([np.array(file_names).reshape((-1, 1)), np.ones((len(file_names), len(y_columns)), dtype=np.uint8)])

    test_df = pd.DataFrame(np_dataset, columns=['image_name'] + y_columns)

    train_df, _ = stratify_shuffle_split_subsets(test_df, train_ratio)

    assert train_df.shape[0] == expected_train_size
