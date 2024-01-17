import pytest
import pandas as pd
import albumentations as albu

from src.dataset import PlanetDataset


@pytest.fixture
def labels_df():
    return pd.DataFrame.from_dict(
        {
            'image_name': ['red_rectangle.jpg', 'red_rectangle.jpg'],
            'clear': [0, 1],
            'partly_cloudy': [0, 0],
            'cloudy': [0, 1],
            'haze': [0, 0],
            'agriculture': [0, 1],
            'cultivation': [0, 0],
            'bare_ground': [0, 1],
            'conventional_mine': [0, 1],
            'artisinal_mine': [0, 0],
            'primary': [0, 1],
            'blooming': [0, 0],
            'selective_logging': [0, 1],
            'blow_down': [0, 0],
            'slash_burn': [0, 1],
            'habitation': [0, 1],
            'water': [0, 0],
            'road': [0, 1],
        }
    )


def test_len_method(labels_df: pd.DataFrame):
    dataset = PlanetDataset(
        labels_df=labels_df,
        images_folder='data/tests',
        transforms=None,
    )

    assert len(dataset) == len(labels_df)


def test_rgb_image(labels_df: pd.DataFrame):
    dataset = PlanetDataset(
        labels_df=labels_df,
        images_folder='data/tests',
        transforms=None,
    )

    red_rect_object = dataset[0][0]

    assert red_rect_object[0][0][0] != 0
    assert red_rect_object[0][0][1] == 0
    assert red_rect_object[0][0][2] == 0


@pytest.mark.parametrize(
    ('transforms', 'expected_size'),
    [
        (albu.Resize(100, 150, always_apply=True), (100, 150, 3)),
        (albu.Compose([albu.ToGray(True), albu.Resize(80, 100, always_apply=True)]), (80, 100, 3)),
    ]
)
def test_augmentations(labels_df: pd.DataFrame, transforms: albu.BasicTransform, expected_size: tuple[int, int, int]):
    dataset = PlanetDataset(
        labels_df=labels_df,
        images_folder='data/tests',
        transforms=transforms,
    )

    red_rect_object = dataset[0][0]

    assert red_rect_object.shape == expected_size
