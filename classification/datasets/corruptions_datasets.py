
import os
import json
import torch
import random
from glob import glob
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    corruptions: Sequence[str] = CORRUPTIONS,
    transform = None,
    setting: str = 'continual'):

    assert len(corruptions) == 1, "so far only one corruption is supported"

    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions = CORRUPTIONS if "non_stationary" in setting else corruptions

    for cor in corruptions:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(n_examples=n_examples,
                                         severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(n_examples=n_examples,
                                          severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i]] for i in range(x_test.shape[0])]
    if "non_stationary" in setting:
        random.shuffle(samples)

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    corruptions: Sequence[str] = CORRUPTIONS,
    transform = None,
    setting: str = 'continual'):

    assert len(corruptions) == 1, "so far only one corruption is supported"

    # load imagenet class to id mapping from robustbench
    with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
        class_to_idx = json.load(f)

    corruptions = CORRUPTIONS if "non_stationary" in setting else corruptions

    # load default 5000 test samples from robust bench
    corruption_dir_path = os.path.join(data_dir, corruptions[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)
    dataset_test.samples = dataset_test.samples[:n_examples]

    if "non_stationary" in setting:
        files = []
        for cor in corruptions:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            file_paths = glob(os.path.join(corruption_dir_path, "*", "*.JPEG"))
            tmp_files = [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]
            files += random.sample(tmp_files, k=min(n_examples, len(tmp_files)))
        random.shuffle(files)
        dataset_test.samples = files
    elif setting == "correlated" or n_examples > 5000:
        # get all test samples of the specified corruption
        file_paths = glob(os.path.join(str(corruption_dir_path), "*", "*.JPEG"))
        files = [(fp, class_to_idx[fp[len(str(corruption_dir_path))+1:].split(os.sep)[0]]) for fp in file_paths]

        if setting != "correlated":
            #  randomly subsample the data to length 'n_examples'
            dataset_test.samples = random.sample(files, k=min(n_examples, len(files)))

    return dataset_test
