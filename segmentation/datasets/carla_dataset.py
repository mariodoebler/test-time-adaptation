
import os
import torch
import random
import logging
import numpy as np
from torch.utils import data
from PIL import Image
from augmentations import augmentations as augs

logger = logging.getLogger(__name__)

# Mean to substract from images (BGR order)
IMG_MEAN = np.array([104.00698793, 116.66876762, 122.67891434])


def create_carla_loader(data_dir, list_path, ignore_label, test_size, batch_size, percentage=1.0, workers=4, transform_train=None, is_training=False):
    dataset = CarlaDataset(data_dir, list_path, ignore_label, test_size=test_size, transform_train=transform_train, is_training=is_training)

    if percentage < 1.0 and is_training is True:
        nr_samples = len(dataset.img_file_paths)
        nr_reduced = int(np.ceil(nr_samples * percentage))
        dataset.img_file_paths = random.sample(dataset.img_file_paths, nr_reduced)
        logger.info(f"Reduced number of images: {nr_reduced}/{nr_samples} \t Reduction factor = {nr_reduced / nr_samples:.5f}")

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       shuffle=True if is_training else False,
                                       drop_last=True if is_training else False)


class CarlaDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 list_name,
                 ignore_label=255,
                 test_size=1024,
                 transform_train=None,
                 is_training=False
                 ):

        self.data_dir = data_dir
        self.list_path = os.path.join("datasets", "carla_lists", list_name)
        self.ignore_label = ignore_label
        self.is_training = is_training

        if not (os.path.exists(self.list_path)):
            raise ValueError(f'List path {self.list_path} does not exist!')

        # open the corresponding file and create a list containing the image paths
        with open(self.list_path) as f:
            self.img_file_paths = [os.path.join(data_dir, id.strip()) for id in f]
        logger.info(f'Found {len(self.img_file_paths)} unique files in {self.list_path}')

        # setup a training and test transformation
        self.transform_train = transform_train
        self.transform_test = augs.Compose([augs.Resize(test_size)])

        self.id_to_trainid = {7: 0, 8: 1, 1: 2, 11: 3, 2: 4, 5: 5, 18: 6, 12: 7,
                              9: 8, 22: 9, 13: 10, 4: 11, 10: 12, 6: 13}

    def __getitem__(self, index):
        img_path = self.img_file_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(img_path.replace('camera', 'segmentation'))

        # Prepare image and label
        if self.is_training and self.transform_train is not None:
            image, label = self.transform_train(image, label)
        else:
            image, label = self.transform_test(image, label)

        # normalize image and convert training ids
        image = self.process_img(image)
        label = self.process_label(label)

        return image, label, img_path.split(os.sep)[-1][:-4]

    def __len__(self):
        return len(self.img_file_paths)

    def id2trainId(self, label):
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def process_img(self, image):
        image = np.asarray(image, np.float32)
        image = torch.from_numpy(image.transpose((2, 0, 1)) / 255.)  # normalized RGB image in range [0, 1]
        return image

    def process_label(self, label):
        target = np.asarray(label, np.float32)
        target = target[:, :, 0]    # In CARLA, the label information is contained in the "red channel"
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)
        return target


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.visualization import batch2image
    from conf import cfg, load_cfg_fom_args
    from augmentations.transformations import get_src_transform

    load_cfg_fom_args("Test_dataset")
    transform_train = get_src_transform(cfg, IMG_MEAN)
    data_loader = create_carla_loader(data_dir=cfg.DATA_DIR,
                                      list_path=cfg.LIST_NAME_TEST,
                                      ignore_label=cfg.OPTIM.IGNORE_LABEL,
                                      test_size=cfg.TEST.IMG_SIZE,
                                      batch_size=cfg.TEST.BATCH_SIZE,
                                      percentage=cfg.SOURCE.PERCENTAGE,
                                      workers=cfg.OPTIM.WORKERS,
                                      transform_train=transform_train,
                                      is_training=False)

    for (imgs, labels, file_ids) in tqdm(data_loader, total=len(data_loader)):
        img_arr, label_arr = batch2image(imgs, labels, nrow=max(imgs.shape[0] // 2, 1))
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(img_arr)
        ax[1].imshow(img_arr)
        ax[1].imshow(label_arr, alpha=0.7)
        plt.show()
