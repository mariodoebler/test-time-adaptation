from copy import deepcopy

import torch
import torch.nn as nn
from torchvision.transforms import Compose, RandomCrop


class TTAMethod(nn.Module):
    """
    """
    def __init__(self, model, optimizer, crop_size, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.rand_crop = Compose([RandomCrop(crop_size[::-1])])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            x_new = torch.cat([self.rand_crop(x.clone()) for _ in range(2)], dim=0)
            _ = self.forward_and_adapt(x_new)

        return self.model(x)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        """
        raise NotImplementedError

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters of the backbone and the segmentation head.

        Note: other choices of parameterization are possible!
        """
        params_feat = []
        params_head = []
        for name, par in model.named_parameters():
            if 'seg_head' not in name and par.requires_grad:
                params_feat.append(par)
            elif 'seg_head' in name and par.requires_grad:
                params_head.append(par)

        return params_feat, params_head
