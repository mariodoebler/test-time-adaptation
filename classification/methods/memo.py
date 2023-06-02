"""
Builds upon: https://github.com/zhangmarvin/memo
Corresponding paper: https://arxiv.org/abs/2110.09506
"""

import numpy as np
from PIL import Image

import torch
import torch.jit

from methods.base import TTAMethod
from augmentations.transforms_memo_cifar import aug_cifar
from augmentations.transforms_memo_imagenet import aug_imagenet


def tta(image, n_augmentations, aug):

    image = np.clip(image[0].cpu().numpy() * 255., 0, 255).astype(np.uint8).transpose(1, 2, 0)
    inputs = [aug(Image.fromarray(image)) for _ in range(n_augmentations)]
    inputs = torch.stack(inputs).cuda()
    return inputs


class MEMO(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.alpha_bn = cfg.BN.ALPHA
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.augmentations = aug_cifar if "cifar" in self.dataset_name else aug_imagenet

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            x_aug = tta(x, self.n_augmentations, aug=self.augmentations)
            _ = self.forward_and_adapt(x_aug)

        return self.model(x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss, _ = marginal_entropy(outputs)
        loss.backward()
        self.optimizer.step()
        return outputs

    def configure_model(self):
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.alpha_bn)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
