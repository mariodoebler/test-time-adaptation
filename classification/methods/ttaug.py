import numpy as np
from PIL import Image

import torch
from copy import deepcopy
from methods.base import TTAMethod
from augmentations.transforms_memo_cifar import aug_cifar
from augmentations.transforms_memo_imagenet import aug_imagenet


def tta(image, n_augmentations, aug):

    image = np.clip(image[0].cpu().numpy() * 255., 0, 255).astype(np.uint8).transpose(1, 2, 0)
    inputs = [aug(Image.fromarray(image)) for _ in range(n_augmentations)]
    inputs = torch.stack(inputs).cuda()
    return inputs


class TTAug(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.alpha_bn = cfg.BN.ALPHA
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.augmentations = aug_cifar if "cifar" in self.dataset_name else aug_imagenet
        self.model_state, _ = self.copy_model_and_optimizer()

    @torch.no_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()

        x_aug = tta(x, self.n_augmentations, aug=self.augmentations)
        outputs = self.model(x_aug).mean(0, keepdim=True)

        return outputs

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def configure_model(self):
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.alpha_bn)
        self.model.requires_grad_(False)
