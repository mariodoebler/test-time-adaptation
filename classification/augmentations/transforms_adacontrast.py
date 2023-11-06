import random
import logging
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentation(aug_type, res_size=256, crop_size=224):
    if aug_type == "moco-v2":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v2-light":
        transform_list = [
            transforms.Resize(res_size),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v1":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain":
        transform_list = [
            transforms.Resize(res_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "clip_inference":
        transform_list = [
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "test":
        transform_list = [
            transforms.Resize(res_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    else:
        return None

    return transforms.Compose(transform_list)


def get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(aug_type, res_size=res_size, crop_size=crop_size))
        elif version == "w":
            transform_list.append(get_augmentation("plain", res_size=res_size, crop_size=crop_size))
        elif version == "t":
            transform_list.append(get_augmentation("test", res_size=res_size, crop_size=crop_size))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform
