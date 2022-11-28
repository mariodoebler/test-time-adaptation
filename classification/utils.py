
import torch
import logging
from datasets.imagenet200_dataset import IMAGENET_R_MASK, IMAGENET_A_MASK

logger = logging.getLogger(__name__)


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device = None,
                 dataset_name: str = "cifar10"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            if dataset_name == "imagenet_a":
                output = output[:, IMAGENET_A_MASK]

            correct += (output.max(1)[1] == labels.to(device)).float().sum()
    return correct.item() / len(data_loader.dataset)
