
import logging
import torch
from torch.utils import model_zoo
from torch import Tensor
from typing import Tuple
from models import deeplabv2, deeplabv2_asm, deeplabv2_smppm

logger = logging.getLogger(__name__)


class ImageNormalizer(torch.nn.Module):
    def __init__(self, mean: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, list):
            input[0] = input[0][:, [2, 1, 0], :, :] * 255. - self.mean  # needed, since smppm receives multiple inputs
        else:
            input = input[:, [2, 1, 0], :, :] * 255. - self.mean
        return input


def load_model(method, ckpt_path, device, img_mean, imagenet_init=True, num_classes=14, model_name="deeplabv2"):
    if model_name == "deeplabv2":
        # create the input normalization which will be later wrapped into the model
        normalize = ImageNormalizer(img_mean).to(device)

        # setup the corresponding model
        if method == 'asm':
            model = deeplabv2_asm.ResNet(deeplabv2_asm.Bottleneck, [3, 4, 23, 3], num_classes).to(device)
        elif method == 'sm_ppm':
            model = deeplabv2_smppm.ResNet(deeplabv2_smppm.Bottleneck, [3, 4, 23, 3], num_classes).to(device)
        else:
            model = deeplabv2.ResNetMulti(deeplabv2.Bottleneck, [3, 4, 23, 3], num_classes).to(device)

        # initialize the model
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            logger.info(f'Successfully restored segmentation model from: {ckpt_path}')
        elif imagenet_init:
            path = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
            saved_state_dict = model_zoo.load_url(path)
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
            logger.info(f'Successfully restored pre-trained ImageNet model from {path}')
        else:
            logger.warning("Model is trained from scratch! This may degrade the final performance!")
    else:
        raise ValueError(f"Model '{model_name}' is not supplied!")

    # add the input pre-processing to the model
    model = torch.nn.Sequential(normalize, model)
    return model.to(device)
