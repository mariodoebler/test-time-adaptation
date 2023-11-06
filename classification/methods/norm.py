import torch.nn as nn
from methods.source import Source
from methods.bn import AlphaBatchNorm, EMABatchNorm
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class BNTest(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        for m in self.model.modules():
            # Re-activate batchnorm layer
            if (isinstance(m, nn.BatchNorm1d) and self.batch_size > 1) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()


@ADAPTATION_REGISTRY.register()
class BNAlpha(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        # (1-alpha) * src_stats + alpha * test_stats
        self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).to(self.device)


@ADAPTATION_REGISTRY.register()
class BNEMA(Source):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = EMABatchNorm.adapt_model(self.model).to(self.device)
