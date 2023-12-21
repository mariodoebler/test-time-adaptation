import os
import logging
import random
import numpy as np
import time
import webdataset as wds

import torch
import torchvision
import torchvision.transforms as transforms

from typing import Union
from conf import complete_data_dir_path
from datasets.imagelist_dataset import ImageList
from datasets.imagenet_subsets import create_imagenet_subset
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset
from datasets.imagenet_d_utils import create_symlinks_and_get_imagenet_visda_mapping
from datasets.imagenet_dict import map_dict
from augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation


logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_transform(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None]):
    """
    Get the transformation pipeline
    Note that the data normalization is done within the model
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
    Returns:
        transforms: The data pre-processing (and augmentation)
    """
    if adaptation == "adacontrast":
        # adacontrast requires specific transformations
        if dataset_name in ["cifar10", "cifar100", "cifar10_c", "cifar100_c"]:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(32, 32), crop_size=32)
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(224, 224), crop_size=224)
        elif dataset_name == "domainnet126":
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=(256, 256), crop_size=224)
        else:
            resize_size = 256
            crop_size = 224
            # try to get the correct resize & crop size from the pre-process
            if isinstance(preprocess, transforms.Compose):
                for transf in preprocess.transforms:
                    if isinstance(transf, transforms.Resize):
                        resize_size = transf.size
                    elif isinstance(transf, (transforms.CenterCrop, transforms.RandomCrop, transforms.RandomResizedCrop)):
                        crop_size = transf.size

            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=resize_size, crop_size=crop_size)
    else:
        # create non-method specific transformation
        if dataset_name in ["cifar10", "cifar100"]:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in ["cifar10_c", "cifar100_c"]:
            transform = None
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == "domainnet126":
            transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
        else:
            if preprocess:
                # set transform to the corresponding input transformation of the restored model
                transform = preprocess
            else:
                # use classical ImageNet transformation procedure
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])

    return transform


def get_test_loader(setting: str, adaptation: str, dataset_name: str, preprocess: Union[transforms.Compose, None],
                    data_root_dir: str, domain_name: str, domain_names_all: list, severity: int, num_examples: int,
                    rng_seed: int, delta_dirichlet: float = 0., batch_size: int = 128, shuffle: bool = False, workers: int = 4):
    """
    Create the test data loader
    Input:
        setting: Name of the considered setting
        adaptation: Name of the adaptation method
        dataset_name: Name of the dataset
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        domain_name: Name of the current domain
        domain_names_all: List containing all domains
        severity: Severity level in case of corrupted data
        num_examples: Number of test samples for the current domain
        rng_seed: A seed number
        delta_dirichlet: Parameter of the Dirichlet distribution
        batch_size: The number of samples to process in each iteration
        shuffle: Whether to shuffle the data. Will destroy pre-defined settings
        workers: Number of workers used for data loading
    Returns:
        test_loader: The test data loader
    """

    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    data_dir = complete_data_dir_path(data_root_dir, dataset_name)
    transform = get_transform(dataset_name, adaptation, preprocess)

    # create the test dataset
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, adaptation, preprocess, data_root_dir,
                                            batch_size, train_split=False, workers=workers)
    else:
        if dataset_name in ["cifar10_c", "cifar100_c"]:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruption=domain_name,
                                                 corruptions_seq=domain_names_all,
                                                 transform=transform,
                                                 setting=setting)

        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform,
                                                    setting=setting)

        elif dataset_name in ["imagenet_k", "imagenet_r", "imagenet_a", "imagenet_v2"]:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

        elif dataset_name == "ccc":
            # url = os.path.join(dset_path, "serial_{{00000..99999}}.tar") Uncoment this to use a local copy of CCC
            # domain_name = "baseline_20_transition+speed_1000_seed_44" # choose from: baseline_<0/20/40>_transition+speed_<1000/2000/5000>_seed_<43/44/45>
            url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{domain_name}/serial_{{00000..99999}}.tar'
            test_dataset = (wds.WebDataset(url)
                    .decode("pil")
                    .to_tuple("input.jpg", "output.cls")
                    .map_tuple(transform, identity)
            )
        elif dataset_name in ["imagenet_d", "imagenet_d109", "domainnet126"]:
            # create the symlinks needed for imagenet-d variants
            if dataset_name in ["imagenet_d", "imagenet_d109"]:
                for dom_name in domain_names_all:
                    if not os.path.exists(os.path.join(data_dir, dom_name)):
                        logger.info(f"Creating symbolical links for ImageNet-D {dom_name}...")
                        domainnet_dir = os.path.join(complete_data_dir_path(data_root_dir, "domainnet126"), dom_name)
                        create_symlinks_and_get_imagenet_visda_mapping(domainnet_dir, map_dict)

            # prepare a list containing all paths of the image-label-files
            if "mixed_domains" in setting:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", dom_name + "_list.txt") for dom_name in domain_names_all]
            else:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", domain_name + "_list.txt")]

            test_dataset = ImageList(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)

        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.shuffle(test_dataset.samples)

        # randomly subsample the dataset if num_examples is specified
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        # prepare samples with respect to the considered setting
        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")

        if "correlated" in setting:
            # sort the file paths by label
            if delta_dirichlet > 0.:
                logger.info(f"Using Dirichlet distribution with delta={delta_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(delta_dirichlet, samples=test_dataset.samples)
            else:
                # sort the class labels by ascending order
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def get_source_loader(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None],
                      data_root_dir: str, batch_size: int, train_split: bool = True, ckpt_path: str = None,
                      num_samples: int = -1, percentage: float = 1.0, workers: int = 4):
    """
    Create the source data loader
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        batch_size: The number of samples to process in each iteration
        train_split: Whether to use the training or validation split
        ckpt_path: Path to a checkpoint which determines the source domain for DomainNet-126
        num_samples: Number of source samples used during training
        percentage: (0, 1] Percentage of source samples used during training
        workers: Number of workers used for data loading
    Returns:
        source_dataset: The source dataset
        source_loader: The source data loader
    """

    # create the correct source dataset name
    src_dataset_name = dataset_name.split("_")[0] if dataset_name != "ccc" else "imagenet"

    # complete the data root path to the full dataset path
    data_dir = complete_data_dir_path(data_root_dir, dataset_name=src_dataset_name)

    # get the data transformation
    transform = get_transform(src_dataset_name, adaptation, preprocess)

    # create the source dataset
    if dataset_name in ["cifar10", "cifar10_c"]:
        source_dataset = torchvision.datasets.CIFAR10(root=data_root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name in ["cifar100", "cifar100_c"]:
        source_dataset = torchvision.datasets.CIFAR100(root=data_root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name in ["imagenet", "imagenet_c", "imagenet_k", "ccc"]:
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in ["domainnet126"]:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("datasets", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        logger.info(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name in ["imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_d", "imagenet_d109"]:
        split = "train" if train_split else "val"
        source_dataset = create_imagenet_subset(data_dir=data_dir,
                                                test_dataset_name=dataset_name,
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples >= 0:    # reduce the number of source samples
        assert percentage > 0.0, "The percentage of source samples has to be in range 0.0 < percentage <= 1.0"
        assert num_samples > 0, "The number of source samples has to be at least 1"
        if src_dataset_name in ["cifar10", "cifar100"]:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader


def sort_by_dirichlet(delta_dirichlet: float, samples: list):
    """
    Adapted from: https://github.com/TaesikGong/NOTE/blob/main/learner/dnn.py
    Sort classes according to a dirichlet distribution
    Input:
        delta_dirichlet: Parameter of the distribution
        samples: List containing all data sample pairs (file_path, class_label)
    Returns:
        samples_sorted: List containing the temporally correlated samples
    """

    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes

    time_start = time.time()
    time_duration = 120  # seconds until program terminates if no solution was found

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:  # prevent any chunk having too less data
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  # contains data per each class
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(delta_dirichlet, dirichlet_numchunks))

            # balance
            proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

            # store class-wise data
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)

        # exit loop if no solution was found after a certain while
        if time.time() > time_start + time_duration:
            raise ValueError(f"Could not correlated sequence using dirichlet value '{delta_dirichlet}'. Try other value!")

    sequence_stats = []

    # create temporally correlated sequence
    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])
            sequence_stats.extend(list(np.repeat(cls, len(idx))))

    return samples_sorted
