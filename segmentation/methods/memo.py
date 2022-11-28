import torch
import torch.jit
from PIL import Image
import numpy as np

from methods.base import TTAMethod
from augmentations.transforms_memo import aug


def tta(image, n_augmentations, aug):
    assert image.shape[0] == 1
    image = (image[0] * 255.).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    inputs = [aug(Image.fromarray(image)) for _ in range(n_augmentations)]
    inputs = torch.stack(inputs).cuda()
    return inputs


class MEMO(TTAMethod):
    """MEMO
    """
    def __init__(self, model, optimizer, crop_size, steps, episodic, n_augmentations):
        super().__init__(model, optimizer, crop_size, steps, episodic)
        self.n_augmentations = n_augmentations

    def forward(self, x):
        if self.episodic:
            self.reset()

        self.model.train()
        for _ in range(self.steps):
            self.forward_and_adapt(x.clone())

        return self.model(x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        bs = 2
        for _ in range(self.n_augmentations // bs):
            x_aug = tta(x, n_augmentations=bs, aug=aug)

            self.optimizer.zero_grad()
            outputs = self.model(x_aug)
            loss, _ = marginal_entropy(outputs)
            loss.backward()

        self.optimizer.step()


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=1, keepdim=True)
    avg_logits = logits.logsumexp(dim=(0, 2, 3)) - np.log(logits.shape[0]) - np.log(logits.shape[2]) - np.log(logits.shape[3])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
