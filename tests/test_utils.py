import pytest
from typing import Any

from src.utils import load_object
from src.datamodule import PlanetDM
from src.augmentations import get_transforms
from src.config import DataConfig


@pytest.mark.parametrize(
    ('input_str', 'expected_obj'),
    [
        ('src.datamodule.PlanetDM', PlanetDM),
        ('src.augmentations.get_transforms', get_transforms),
        ('src.config.DataConfig', DataConfig),
    ]
)
def test_load_object_function(input_str: str, expected_obj: Any):
    test_object = load_object(input_str)

    assert test_object == expected_obj
