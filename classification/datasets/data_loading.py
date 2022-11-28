import os
import logging
import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from conf import complete_data_dir_path, get_domain_sequence
from datasets.imagelist_dataset import ImageList
from datasets.imagenet200_dataset import create_imagenet200_dataset
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset
from augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation


logger = logging.getLogger(__name__)


def get_transform(dataset_name, adaptation):
    """
    Get transformation pipeline
    Note that the data normalization is done inside of the model
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """
    if adaptation == "adacontrast":
        # adacontrast requires specific transformations
        if dataset_name in {"cifar10", "cifar100", "cifar10_c", "cifar100_c"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=32, crop_size=32)
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized an centre cropped
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=224, crop_size=224)
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224)
        else:
            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="iwss", aug_type="moco-v2", res_size=256, crop_size=224)
    else:
        # create non-method specific transformation
        if dataset_name in {"cifar10", "cifar100"}:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"cifar10_c", "cifar100_c"}:
            transform = None
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized an centre cropped
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation(aug_type="test", res_size=256, crop_size=224)
        else:
            # use classical ImageNet transformation procedure
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    return transform


def get_test_loader(setting, adaptation, dataset_name, root_dir, domain_name, severity, num_examples, ckpt_path=None, batch_size=128, shuffle=False, workers=4):
    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(1)
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)

    # create the test dataset
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=False)
    else:
        if dataset_name in {"cifar10_c", "cifar100_c"}:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 n_examples=num_examples,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruptions=[domain_name],
                                                 transform=transform,
                                                 setting=setting)
        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruptions=[domain_name],
                                                    transform=transform,
                                                    setting=setting)
        elif dataset_name in {"imagenet_r", "imagenet_a"}:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        elif dataset_name in {"domainnet126"}:
            if "non_stationary" in setting:
                domain_seq = get_domain_sequence(ckpt_path)
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", dom + "_list.txt") for dom in domain_seq]
            else:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", domain_name + "_list.txt")]

            test_dataset = ImageList(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    if "correlated" in setting:
        if hasattr(test_dataset, 'samples'):
            # sort the file paths by label
            test_dataset.samples.sort(key=lambda x: x[1])
        else:
            raise ValueError(f"The setting '{setting}' is not supported for: {dataset_name} {domain_name}")
    elif dataset_name in {"imagenet_r"} or domain_name == "none":
        # shuffle the data since it is sorted by class
        if hasattr(test_dataset, 'samples'):
            random.shuffle(test_dataset.samples)

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=True, ckpt_path=None, num_samples=None, percentage=1.0, workers=4):
    # setup the transformation pipeline
    dataset_name = dataset_name[:-2] if dataset_name in {"cifar10_c", "cifar100_c", "imagenet_c"} else dataset_name
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)

    # create source dataset
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name == "imagenet":
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in {"domainnet126"}:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("datasets", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        logger.info(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name == "imagenet_r":
        split = "train" if train_split else "val"
        data_dir = complete_data_dir_path(root=root_dir, dataset_name="imagenet")
        source_dataset = create_imagenet200_dataset(data_dir=data_dir,
                                                    dataset_name=dataset_name,
                                                    split=split,
                                                    transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    # reduce the number of source samples
    if percentage < 1.0 or num_samples:    # reduce the number of source samples
        if dataset_name in {"cifar10", "cifar100"}:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of source batches in source loader: {len(source_loader)}")
    return source_dataset, source_loader
