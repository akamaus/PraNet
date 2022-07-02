import logging
from glob import glob
from typing import Union

import cv2
import torch


class TransformingDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform


class MultiGlobDataset(TransformingDataset):
    def __init__(self, patterns: Union[str, list[str]], skiplist=None):
        super().__init__()

        if skiplist is None:
            skiplist = []

        if isinstance(patterns, str):
            patterns = [patterns]

        files = []
        for pat in patterns:
            files += list(glob(pat))

        self.files = [f for f in files if f not in set(skiplist)]

    def __iter__(self):
        for fn in self.files:
            try:
                if self.transform is None:
                    data = cv2.imread(fn)
                    yield {'image': torch.tensor(data[:, :, ::-1].copy()) / 255.0, 'path': fn}
                else:
                    sample = self.transform({'img_info': {'filename': fn}, 'img_prefix': None})
                    logging.debug('ret sample')
                    yield sample
            except (AttributeError, RuntimeError) as e:
                logging.warning(f'Got {e} during processing {fn}')