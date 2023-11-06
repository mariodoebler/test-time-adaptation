import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional

logger = logging.getLogger(__name__)


class ImageList(Dataset):
    def __init__(self, image_root: str, label_files: Sequence[str], transform: Optional[Callable] = None):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        """Build a list of <image path, class label, domain name> items.
        Input:
            label_file: Path to the file containing the image label pairs
        Returns:
            item_list: A list of <image path, class label> items.
        """
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain_name))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path
