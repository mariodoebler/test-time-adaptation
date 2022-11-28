
import wandb
import torch
import logging
import numpy as np
import torch.nn.functional as F
from utils.func import fast_hist, per_class_iou, print_per_class_iou, CLASS_NAMES_CARLA
from utils.visualization import save_col_preds

logger = logging.getLogger(__name__)


def evaluate_sequence(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      num_classes: int,
                      save_preds: bool,
                      preds_dir_path: str):

    miou = 0.
    confusion_matrix = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for i, (imgs, labels, file_ids) in enumerate(data_loader):
            logits = model(imgs.to(device))
            logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=True)
            _, predictions = torch.max(logits.softmax(dim=1), dim=1)

            # save the predictions
            if save_preds:
                # entropy_maps = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
                save_col_preds(preds_dir_path, file_ids, predictions, entropy_maps=None, images=imgs.clone())

            # calculate the mean-intersection-over-union (miou)
            predictions = predictions.squeeze(0).cpu().numpy()
            labels = labels.squeeze(0).cpu().numpy()
            confusion_matrix += fast_hist(labels.flatten(), predictions.flatten(), num_classes)

            # calculate the mIoU for all encountered samples
            iou_classes_total = per_class_iou(confusion_matrix)
            miou = round(np.nanmean(iou_classes_total) * 100, 2)

            # log to wandb
            wandb.log({'mIoU_accumulated': miou}, step=i)

    print_per_class_iou(iou_classes_total, class_names=CLASS_NAMES_CARLA)
    return miou
