import numpy as np

import torch
import torchvision.transforms as transforms

from augmentations import augmix_ops


# AugMix Transforms
def get_preaugment(dataset_name, crop_size=224):
    if "cifar" in dataset_name and crop_size == 32:
        preaugment = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        preaugment = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip()
        ])
    return preaugment


def augmix(image, preprocess, aug_list, dataset_name, severity=1, crop_size=224):
    preaugment = get_preaugment(dataset_name, crop_size)
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity, crop_size)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, dataset_name="imagenet", n_views=2, use_augmix=False, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.dataset_name = dataset_name
        self.n_views = n_views
        self.severity = severity

        self.aug_list = augmix_ops.augmentations if use_augmix else []

        self.img_size = 32 if "cifar" in self.dataset_name else 224
        # restore the final image input size from the base transform
        if isinstance(base_transform, transforms.Compose):
            for transf in base_transform.transforms[::-1]:
                if isinstance(transf, (transforms.Resize, transforms.RandomResizedCrop, transforms.RandomCrop, transforms.CenterCrop)):
                    self.img_size = getattr(transf, "size")
                    self.img_size = self.img_size[0] if isinstance(self.img_size, (list, tuple)) else self.img_size
                    break

    def __call__(self, x):
        views = [self.preprocess(self.base_transform(x))] if self.base_transform else [self.preprocess(x)]
        views += [augmix(x, self.preprocess, self.aug_list, self.dataset_name, self.severity, self.img_size) for _ in range(self.n_views)]
        return views

