import torch
from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return idx


@ADAPTATION_REGISTRY.register()
class VTE(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.selection_p = cfg.TPT.SELECTION_P

    @forward_decorator
    def forward_and_adapt(self, imgs_test):
        # create a batch by concatenating the augmented versions of current test sample
        imgs_test = torch.cat(imgs_test, dim=0)

        logits, img_features, text_features = self.model(imgs_test, return_features=True)

        idx_confident = select_confident_samples(logits, self.selection_p)

        # ensemble the most confident image features
        img_features_avg = img_features[idx_confident].mean(dim=0, keepdim=True)

        # create the output prediction using the ensembled image features
        output = self.model.logit_scale.exp() * img_features_avg @ text_features.t()
        return output

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def configure_model(self):
        """Configure model."""
        self.model.eval()
        self.model.requires_grad_(False)
