import os
from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch import Tensor

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class PlanetDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        """Inits Planet dataset

        Args:
            labels_df (pd.DataFrame): labels for the images in the specified folder
            images_folder (str): directory where images were saved
            transforms (Optional[TRANSFORM_TYPE], optional): sequence of transforms that should be applied. Defaults to None.
        """
        self.labels_df = labels_df
        self.images_folder = images_folder
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[Tensor, np.array]:
        """Returns image and labels

        Args:
            idx (int): index of the object

        Returns:
            tuple[Tensor, np.array]: image as Tensor and labels in numpy array 
        """
        data_row = self.labels_df.iloc[idx]

        image_path = os.path.join(self.images_folder, data_row['image_name'])
        labels = np.array(data_row[1:], dtype=np.int8)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {'image': image, 'labels': labels}

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['labels']

    def __len__(self) -> int:
        """Returns length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.labels_df)
