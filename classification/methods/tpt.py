import torch
import numpy as np
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


@ADAPTATION_REGISTRY.register()
class TPT(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.selection_p = cfg.TPT.SELECTION_P
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    def forward(self, x):
        if self.episodic:
            # reset prompt and optimizer
            self.model.reset()
            self.optimizer.load_state_dict(self.optimizer_state)

        # create a batch by concatenating the augmented versions of the current test sample
        x = torch.cat(x, dim=0)
        x = self.model.normalize(x.type(self.model.dtype))

        # get the static image features
        # (placing it outside of forward_and_adapt increases the efficiency when more updates steps are used)
        with torch.cuda.amp.autocast():
            img_features = self.model.image_encoder(x)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        selected_idx = None
        for _ in range(self.steps):
            selected_idx = self.forward_and_adapt(img_features, selected_idx)

        # create the final prediction using the learned prompt
        with torch.cuda.amp.autocast():
            text_features = self.model.get_text_features()
            output = self.model.logit_scale.exp() * img_features[:1] @ text_features.t()

        return output

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, img_features, selected_idx):
        with torch.cuda.amp.autocast():
            text_features = self.model.get_text_features()
            logits = self.model.logit_scale.exp() * img_features @ text_features.t()

            if selected_idx is not None:
                logits = logits[selected_idx]
            else:
                logits, selected_idx = select_confident_samples(logits, self.selection_p)

            loss = avg_entropy(logits)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return selected_idx

    def configure_model(self):
        """Configure model."""
        self.model.eval()
        self.model.requires_grad_(False)

        # re-enable parameters
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and not "token_embedding" in name:
                param.requires_grad_(True)

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and param.requires_grad:
                params.append(param)
                names.append(name)
        return params, names
