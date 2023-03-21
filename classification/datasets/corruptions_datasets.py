
import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset

logger = logging.getLogger(__name__)


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # load imagenet class to id mapping from robustbench
    with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
        class_to_idx = json.load(f)

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting:
        files = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            file_paths = glob(os.path.join(corruption_dir_path, "*", "*.JPEG"))
            files += [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]
        dataset_test.samples = files
    elif setting == "correlated" or n_examples != -1:
        # get all test samples of the specified corruption
        file_paths = glob(os.path.join(str(corruption_dir_path), "*", "*.JPEG"))
        files = [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]
        dataset_test.samples = files

    return dataset_test
