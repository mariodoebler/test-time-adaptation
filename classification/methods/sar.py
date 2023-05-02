"""
Builds upon: https://github.com/mr-eggplant/SAR/blob/main/sar.py
Corresponding paper: https://openreview.net/pdf?id=g2YraF75Tj
"""

import torch
import torch.nn as nn
import torch.jit
import numpy as np
import logging

from methods.base import TTAMethod


logger = logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema, new_data, alpha=0.9):
    if ema is None:
        return new_data
    else:
        return alpha * ema + (1 - alpha) * new_data


class SAR(TTAMethod):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps, episodic, window_length, margin_e0, reset_constant_em=0.2):
        super().__init__(model.cuda(), optimizer, steps, episodic, window_length)

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        self.optimizer.zero_grad()
        outputs = self.model(imgs_test)

        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(self.model(imgs_test))
        entropys2 = entropys2[filter_ids_1]  # second time forward
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item())  # record moving average loss values for model recovery
        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        if self.ema is not None:
            if self.ema < self.reset_constant_em:
                logger.info(f"ema < {self.reset_constant_em}, now reset the model")
                self.reset()

        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()
        self.ema = None

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    @staticmethod
    def configure_model(model):
        """Configure model for use with SAR."""
        # model.train()
        model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what SAR updates
        model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model

    @staticmethod
    def check_model(model):
        """Check model for compatability with SAR."""
        is_training = model.training
        assert is_training, "SAR needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "SAR needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "SAR should not update all params: " \
                                   "check which require grad"
        has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
        assert has_norm, "SAR needs normalization layer parameters for its optimization"


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
