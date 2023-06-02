"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""

import torch
import torch.nn as nn
import torch.jit

from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.mt = cfg.M_TEACHER.MOMENTUM
        self.rst = cfg.COTTA.RST
        self.ap = cfg.COTTA.AP
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS

        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = self.copy_model(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(self.dataset_name)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        # Create the prediction of the anchor (source) model
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(imgs_test), dim=1).max(1)[0]

        # Augmentation-averaged Prediction
        outputs_emas = []
        if anchor_prob.mean(0) < self.ap:
            for _ in range(self.n_augmentations):
                outputs_ = self.model_ema(self.transform(imgs_test)).detach()
                outputs_emas.append(outputs_)

            # Threshold choice discussed in supplementary
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            # Create the prediction of the teacher model
            outputs_ema = self.model_ema(imgs_test)

        # Student update
        loss = (self.softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        if self.rst > 0.:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model_ema(imgs_test)

    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)


@torch.jit.script
def softmax_entropy_cifar(x, x_ema):# -> torch.Tensor: 
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema):# -> torch.Tensor:       
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1) 
