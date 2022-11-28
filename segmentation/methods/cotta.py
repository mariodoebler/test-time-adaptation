"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""

import random
from copy import deepcopy

import torch
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod


@torch.no_grad()
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(TTAMethod):
    """CoTTA
    """
    def __init__(self, model, optimizer, crop_size, steps, episodic, n_augmentations=6, mt_alpha=0.999, rst_m=0.01, ap=0.9):
        super().__init__(model, optimizer, crop_size, steps, episodic)

        self.n_augmentations = n_augmentations
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

        # Setup EMA and anchor (source) model
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = deepcopy(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # define test-time transformations
        scale_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]    # 2.0 corresponds to original test size which is used during testing
        self.augmentation_shapes = [(int(ratio * 0.5 * crop_size[1]), int(ratio * 0.5 * crop_size[0])) for ratio in scale_ratios]

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            x_new = torch.cat([self.rand_crop(x.clone()) for _ in range(2)], dim=0)
            _ = self.forward_and_adapt(x_new)

        return self.model_ema(x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        outputs = self.model(x)

        # Get anchor and teacher prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(dim=1)[0]
        outputs_ema = self.model_ema(x)

        if anchor_prob.mean() < self.ap:
            # Create the augmentation-averaged prediction
            outputs_ema = self.create_ensemble_pred(x, outputs_ema)

        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        if self.rst > 0.:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

    @torch.no_grad()
    def create_ensemble_pred(self, x, ema_pred):
        inp_shape = x.shape[2:]
        for aug_shape in self.augmentation_shapes:
            # augment the input
            flip = [random.random() <= 0.5 for _ in range(x.shape[0])]
            tmp_input = torch.cat([x[i:i+1].flip(dims=(3,)) if fp else x[i:i+1] for i, fp in enumerate(flip)], dim=0)
            tmp_input = F.interpolate(tmp_input, size=aug_shape, mode='bilinear', align_corners=True)

            # predict the input but do not interpolate the prediction to the input size
            tmp_output = self.model_ema([tmp_input, False])

            # undo the augmentation
            tmp_output = torch.cat([tmp_output[i:i + 1].flip(dims=(3,)) if fp else tmp_output[i:i + 1] for i, fp in enumerate(flip)], dim=0)
            ema_pred += F.interpolate(tmp_output, size=inp_shape, mode='bilinear', align_corners=True)

        ema_pred /= len(self.augmentation_shapes) + 1
        return ema_pred


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

