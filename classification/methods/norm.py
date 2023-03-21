import torch
from copy import deepcopy
from methods.base import TTAMethod


class Norm(TTAMethod):
    def __init__(self, model, optimizer, steps, episodic, window_length):
        super().__init__(model.cuda(), optimizer, steps, episodic, window_length)

        self.model_state, _ = self.copy_model_and_optimizer()

    @torch.no_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        return self.model(imgs_test)

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
