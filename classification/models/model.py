import logging

import timm
import torch
import torch.nn as nn
import torchvision.models as models

from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from copy import deepcopy
from models import resnet26
from datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_D109_MASK

logger = logging.getLogger(__name__)


def get_torchvision_model(model_name, weight_version="IMAGENET1K_V1"):
    """
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    :param model_name: name of the model to create and initialize with pre-trained weights
    :param weight_version: name of the pre-trained weights to restore
    :return: pre-trained model
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
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights for model '{model_name}' from torchvision!")
    return model


def get_timm_model(model_name):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    :param model_name: name of the model to create and initialize with pre-trained weights
    :return: pre-trained model
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if not model_name in available_models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # add the corresponding input normalization to the model
    if hasattr(model, "pretrained_cfg"):
        logger.info(f"General model information: {model.pretrained_cfg}")
        logger.info(f"Adding input normalization to the model using: mean={model.pretrained_cfg['mean']} \t std={model.pretrained_cfg['std']}")
        model = normalize_model(model, mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    else:
        raise AttributeError(f"Attribute 'pretrained_cfg' is missing for model '{model_name}' from timm."
                             f" This prevents adding the correct input normalization to the model!")

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
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)

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


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

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


def get_model(cfg, num_classes):
    if cfg.CORRUPTION.DATASET == "domainnet126":
        base_model = ResNetDomainNet126(arch=cfg.MODEL.ARCH, checkpoint_path=cfg.CKPT_PATH, num_classes=num_classes)
    else:
        try:
            # load model from torchvision
            base_model = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)
        except ValueError:
            try:
                # load model from timm
                base_model = get_timm_model(cfg.MODEL.ARCH)
            except ValueError:
                try:
                    # load some custom models
                    if cfg.MODEL.ARCH == "resnet26_gn":
                        base_model = resnet26.build_resnet26()
                        checkpoint = torch.load(cfg.CKPT_PATH, map_location="cpu")
                        base_model.load_state_dict(checkpoint['net'])
                        base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
                    else:
                        raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")
                    logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.CKPT_PATH}")
                except ValueError:
                    # load model from robustbench
                    dataset_name = cfg.CORRUPTION.DATASET.split("_")[0]
                    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset_name, ThreatModel.corruptions)

        if cfg.CORRUPTION.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.CORRUPTION.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.CORRUPTION.DATASET == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)

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
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name == "imagenet_a":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_A_MASK))
    elif dataset_name == "imagenet_r":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_R_MASK))
    elif dataset_name == "imagenet_d109":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_D109_MASK))

    return encoder, classifier
