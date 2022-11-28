# Builds upon: https://github.com/DianCh/AdaContrast/blob/master/image_list.py

import os
import logging
import random
from PIL import Image
from typing import Sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform=None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

        # shuffle the data if several files are loaded
        if len(label_files) > 1:
            random.shuffle(self.samples)

    def build_index(self, label_file):
        """Build a list of <image path, class label> items.
        Args:
            label_file: path to the domain-net label file
        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label, img_file))

        return item_list

    def __getitem__(self, idx):
        """Retrieve data for one item.
        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

