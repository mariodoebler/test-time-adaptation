import logging
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from torchvision import transforms

from copy import deepcopy
from functools import wraps
from models.model import ResNetDomainNet126


logger = logging.getLogger(__name__)


class TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables for resetting the model after a certain amount of performed update steps
        self.performed_updates = 0
        self.reset_after_num_updates = cfg.MODEL.RESET_AFTER_NUM_UPDATES

        # restore the image input size from the model pre-processing if it is defined
        # this is required for methods relying on test-time augmentation
        self.img_size = (32, 32) if "cifar" in self.dataset_name else (224, 224)
        if hasattr(self.model, "model_preprocess") and isinstance(self.model.model_preprocess, transforms.Compose):
            for transf in self.model.model_preprocess.transforms[::-1]:
                if hasattr(transf, "size"):
                    self.img_size = getattr(transf, "size")
                    if self.dataset_name in ["imagenet_c", "ccc"] and max(self.img_size) > 224:
                        raise ValueError(f"The specified model with pre-processing {model.model_preprocess} "
                                         f"is not suited in combination with ImageNet-C and CCC! "
                                         f"These datasets are already resized and center cropped to 224")
                    break

        # configure model and optimizer
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()

        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # setup for mixed-precision or single precision
        self.mixed_precision = cfg.MIXED_PRECISION
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

    def forward(self, x):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)

                    # if specified, reset the model after a certain amount of update steps
                    self.performed_updates += 1
                    if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                        self.reset()

                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)

                # if specified, reset the model after a certain amount of update steps
                self.performed_updates += 1
                if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                    self.reset()

        return outputs

    def loss_calculation(self, x):
        """
        Loss calculation.
        """
        raise NotImplementedError

    def forward_and_adapt(self, x):
        """
        Forward and adapt the model on a batch of data.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'AdamW':
            return torch.optim.AdamW(self.params,
                                     lr=self.cfg.OPTIM.LR,
                                     betas=(self.cfg.OPTIM.BETA, 0.999),
                                     weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

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
    def copy_model(model):
        if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()


def forward_decorator(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs): 
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = fn(self, *args, **kwargs)
        else:
            outputs = fn(self, *args, **kwargs)
        return outputs
    return decorator

