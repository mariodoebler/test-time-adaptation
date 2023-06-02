import torch
import torch.nn as nn
from copy import deepcopy
from methods.base import TTAMethod
from methods.bn import AlphaBatchNorm, EMABatchNorm


class Norm(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

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

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        if self.cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
            for m in self.model.modules():
                # Re-activate batchnorm layer
                if (isinstance(m, nn.BatchNorm1d) and self.batch_size > 1) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
        elif self.cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
            # (1-alpha) * src_stats + alpha * test_stats
            self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).cuda()
        elif self.cfg.MODEL.ADAPTATION == "norm_ema":  # BN--EMA
            self.model = EMABatchNorm.adapt_model(self.model).cuda()