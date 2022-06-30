import logging
from glob import glob
from typing import Union

import cv2
import torch


class TransformingDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform


class MultiGlobDataset(TransformingDataset):
    def __init__(self, patterns: Union[str, list[str]]):
        super().__init__()

        if isinstance(patterns, str):
            patterns = [patterns]

        files = []
        for pat in patterns:
            files += list(glob(pat))

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fn = self.files[item]
        if self.transform is None:
            data = cv2.imread(fn)
            return {'image': torch.tensor(data[:, :, ::-1].copy()) / 255.0, 'path': fn}
        else:
            sample = self.transform({'img_info': {'filename': fn}, 'img_prefix': None})
            return sample