import logging
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from models import resnet26
from datasets.imagenet200_dataset import IMAGENET_R_MASK

logger = logging.getLogger(__name__)

# ImageNet normalization for torchvision pre-trained model
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_torchvision_model(model_name, weight_version="IMAGENET1K_V1"):
    """
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    :param model_name: name of the model to create
    :param weight_version: name of the pre-trained weights to restore
    :return:
    """
    # create a dictionary that maps the model name to the corresponding weight function
    name_to_weights = {name[:-8].lower(): name for name in dir(models) if "Weights" in name}
    if not model_name in name_to_weights.keys():
        raise ValueError(f"Model name '{model_name}' is not supported. Choose from: {name_to_weights.keys()}")

    # get the weight function and check if the specified type of weights is available
    model_weights = getattr(models, name_to_weights[model_name])
    available_weight_versions = [version for version in dir(model_weights) if "IMAGENET1K" in version]
    if not weight_version in available_weight_versions:
        raise ValueError(f"Weight type '{weight_version}' is not supported. Choose from: {available_weight_versions}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the pre-trained weights
    model = getattr(models, model_name)
    model = model(weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights for model '{model_name}'!")
    return model


class ResNetDomainNet126(torch.nn.Module):
    """
    Architecture used for DomainNet-126
    """
    def __init__(self, arch="resnet50", checkpoint_path=None, num_classes=126, bottleneck_dim=256):
        super().__init__()

        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = models.__dict__[self.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = models.__dict__[self.arch](pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim

        self.fc = torch.nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")

        # add input normalization to the model
        self.encoder = nn.Sequential(ImageNormalizer(MEAN, STD), self.encoder)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]
        for name, param in model_state_dict.items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0


class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """
    def __init__(self, model, arch_name, dataset_name):
        super().__init__()

        self.encoder, self.fc = split_up_model(model, arch_name=arch_name, dataset_name=dataset_name)
        if isinstance(self.fc, nn.Sequential):
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    self._num_classes = module.out_features
                    self._output_dim = module.in_features
        elif isinstance(self.fc, nn.Linear):
            self._num_classes = self.fc.out_features
            self._output_dim = self.fc.in_features
        else:
            raise ValueError("Unable to detect output dimensions")

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim


class ImageNetRWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()
        # print(model.__dict__)
        # self.model = model
        self.masking_layer = ImageNetRMaskingLayer()

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


class ImageNetRMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, IMAGENET_R_MASK]


def get_model(cfg, num_classes):
    if cfg.CORRUPTION.DATASET == "domainnet126":
        base_model = ResNetDomainNet126(arch=cfg.MODEL.ARCH, checkpoint_path=cfg.CKPT_PATH, num_classes=num_classes)
    else:
        try:
            # load model from torchvision
            base_model = get_torchvision_model(cfg.MODEL.ARCH, pretrained=True)

        except ValueError:
            try:
                # load some custom models
                if cfg.MODEL.ARCH == "resnet26":
                    base_model = resnet26.build_resnet26()
                    checkpoint = torch.load(cfg.CKPT_PATH, map_location="cpu")
                    base_model.load_state_dict(checkpoint['net'])
                    base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
                else:
                    raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")
                logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.CKPT_PATH}")
            except ValueError:
                # load model from robustbench
                dataset_name = cfg.CORRUPTION.DATASET[:-2] if cfg.CORRUPTION.DATASET in {"imagenet_c", "cifar10_c", "cifar100_c"} else cfg.CORRUPTION.DATASET
                base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset_name, ThreatModel.corruptions)

        if cfg.CORRUPTION.DATASET == "imagenet_r":
            base_model = ImageNetRWrapper(base_model)
    return base_model.cuda()


def split_up_model(model, arch_name, dataset_name):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    :param model: model to be split up
    :param arch_name: name of the network
    :param dataset_name: name of the dataset
    :return: encoder and classifier
    """
    if arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name in {"Hendrycks2020AugMix_ResNeXt"}:
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name in {"imagenet", "imagenet_c"} or arch_name == "resnet26":
        if arch_name == "vit_b_16":
            encoder = TransformerWrapper(model)
            classifier = model.model.heads
        else:
            encoder = nn.Sequential(model.normalize, nn.Sequential(*list(model.model.children())[:-1]), nn.Flatten())
            classifier = model.model.fc
    elif dataset_name == "imagenet_r":
        if arch_name == "vit_b_16":
            encoder = TransformerWrapper(model)
            classifier = nn.Sequential(model.model.heads.head,  ImageNetRMaskingLayer())
        else:
            encoder = nn.Sequential(model.normalize, nn.Sequential(*list(model.model.children())[:-1]), nn.Flatten())
            classifier = nn.Sequential(model.model.fc, ImageNetRMaskingLayer())
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return encoder, classifier
