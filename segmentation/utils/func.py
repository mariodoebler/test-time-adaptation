
import logging
import numpy as np

logger = logging.getLogger(__name__)


"""
Information about CARLA semantic segmentation:
https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
"""
CLASS_NAMES_CARLA = ["road", "sidewalk", "building", "wall", "fence", "pole", "trafficlight", "trafficsign",
                     "vegetation", "terrain", "sky", "person", "vehicle", "roadline"]


def fast_hist(y_true, y_pred, num_classes):
    mask = (y_true >= 0) & (y_true < num_classes)
    return np.bincount(num_classes * y_true[mask].astype(int) + y_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_iou(hist):
    # iou = TP / (TP + FN + FP)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def print_per_class_iou(iou_classes, class_names):
    assert iou_classes.shape[0] == len(class_names), "Shape of 'iou_classes' and 'class_names' does not match!"
    logger.info('{:<15}    iou:'.format('class name'))

    for i in range(iou_classes.shape[0]):
        logger.info('  {:<15}: {:.2f}'.format(class_names[i], iou_classes[i] * 100))
    logger.info('____________________________')
    logger.info('  {:<15}: {}\n'.format('mIoU', round(np.nanmean(iou_classes) * 100, 2)))
