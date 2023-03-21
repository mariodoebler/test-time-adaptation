# Builds upon: https://github.com/DianCh/AdaContrast/blob/master/image_list.py

import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional

logger = logging.getLogger(__name__)


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path

    def __len__(self):
        return len(self.samples)
