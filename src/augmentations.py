from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    """Returns transforms for the dataset

    Args:
        width (int): result image width
        height (int): result image height
        preprocessing (bool, optional): defines if we should resize image. Defaults to True.
        augmentations (bool, optional): defines if we should apply augmentations. Defaults to True.
        postprocessing (bool, optional): defines if we should normalize and convert to Torch tensor. Defaults to True.

    Returns:
        TRANSFORM_TYPE: sequence of transforms
    """
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend([])

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
