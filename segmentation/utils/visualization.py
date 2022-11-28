import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

"""
Information about CARLA semantic segmentation:
https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
"""

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250,
           170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 157, 234, 50]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    else:
        mask = np.asarray(mask, np.uint8)

    col_mask = Image.fromarray(mask).convert('P')
    col_mask.putpalette(palette)
    return np.asarray(col_mask.convert("RGB"))


def batch2image(images=None, labels=None, nrow=1):
    # convert images
    if images is not None:
        images = make_grid(images, nrow=nrow).cpu().numpy()
        images = np.transpose(images, (1, 2, 0))

    # colorize labels and convert them
    if labels is not None:
        col_labels = []
        for i in range(labels.shape[0]):
            col_labels.append(colorize_mask(labels[i].squeeze()))
        labels = torch.tensor(np.transpose(col_labels, (0, 3, 1, 2)))
        labels = make_grid(labels, nrow=nrow).int().numpy()
        labels = np.transpose(labels, (1, 2, 0)).astype(dtype=np.uint8)
    return images, labels


def unit_range(x):
    return (x - x.min()) / (x.max() - x.min())


def save_col_preds(output_dir, file_names, predictions, entropy_maps=None, images=None):
    """
    :param output_dir: Path of output directory
    :param file_names: List containing the file names
    :param predictions: Hard predictions (N, H, W)
    :param entropy_maps: Pixel-wise entropy (N, H, W)
    :param images: the corresponding rgb images (N, C, H, W)
    """
    for i, fname in enumerate(file_names):
        output_path = os.path.join(output_dir, fname + '.jpg')
        imgs_arr, pseudos_arr = batch2image(images=images[i:i+1] if images is not None else None,
                                            labels=predictions[i:i+1].unsqueeze(0),
                                            nrow=1)

        # overlay colorized pseudo-label with image if it was provided
        if imgs_arr is not None:
            if imgs_arr.max() <= 1.0:
                imgs_arr *= 255.
            pseudos_arr = np.asarray(0.3 * imgs_arr + 0.7 * pseudos_arr, dtype=np.uint8)

        # add entropy map to final output image if it was provided
        if entropy_maps is not None:
            entropy = unit_range(entropy_maps[i])
            entropy = entropy.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            entropy = make_grid(entropy, nrow=1)
            entropy = (entropy * 255).type(torch.uint8).cpu().numpy().transpose((1, 2, 0))
            pseudos_arr = np.concatenate([pseudos_arr, entropy], axis=0)

        Image.fromarray(pseudos_arr).save(output_path)
