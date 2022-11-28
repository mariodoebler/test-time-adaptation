
import logging
import torch.nn as nn
from copy import deepcopy


logger = logging.getLogger(__name__)


class NormEMA(nn.Module):
    def __init__(self, model, episodic):
        super().__init__()
        self.model = model
        self.episodic = episodic
        self.model_state_dict = deepcopy(self.model.state_dict())

    def forward(self, x):
        if self.episodic:
            self.reset()

        self.model.train()
        _ = self.model(x)
        self.model.eval()

        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state_dict, strict=True)
