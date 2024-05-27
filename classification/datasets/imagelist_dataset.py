import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional

logger = logging.getLogger(__name__)


class ImageList(Dataset):
    def __init__(self, image_root: str, label_files: Sequence[str], transform: Optional[Callable] = None, split: str = "test"):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            if file.endswith(".json"):
                self.samples += self.build_index_json(label_file=file, split=split)
            else:
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

    def build_index_json(self, label_file, split):
        item_list = []
        with open(label_file) as fp:
            splits = json.load(fp)
            for sample in splits[split]:
                img_path = os.path.join(self.image_root, sample[0])
                item_list.append((img_path, sample[1], split))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path


class FGVCAircraft(Dataset):
    def __init__(self, image_root: str, transform: Optional[Callable] = None, split: str = 'test'):
        self.image_root = image_root
        self.transform = transform

        with open(os.path.join(self.image_root, "variants.txt"), 'r') as fp:
            self.cls_names = [l.replace("\n", "") for l in fp.readlines()]

        self.samples = []
        with open(os.path.join(self.image_root, f'images_variant_{split}.txt'), 'r') as fp:
            lines = [s.strip().split() for s in fp.readlines()]
            for items in lines:
                img_path = os.path.join(self.image_root, "images", f"{items[0]}.jpg")
                label = self.cls_names.index(" ".join(items[1:]))
                self.samples.append((img_path, int(label), split))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, domain, img_path
