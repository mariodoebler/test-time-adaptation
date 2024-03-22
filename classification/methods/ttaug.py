import torch
from copy import deepcopy
from methods.bn import AlphaBatchNorm
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class TTAug(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.model_state, _ = self.copy_model_and_optimizer()

    @forward_decorator
    def forward(self, x):
        x_aug = torch.cat(x, dim=0)
        outputs = self.model(x_aug).mean(0, keepdim=True)
        return outputs

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        return model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def configure_model(self):
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA)
        self.model.requires_grad_(False)
