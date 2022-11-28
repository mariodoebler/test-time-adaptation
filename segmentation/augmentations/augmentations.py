
import torch
import random
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import ColorJitter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask, = t(img, mask)
        return img, mask


class ToTensor(object):
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return img, mask


class Resize(object):
    """
    Adapted from: torchvision.transforms.Resize
    """
    def __init__(self, size):
        self.match_min_size = True
        if isinstance(size, int):
            self.size = size, size
        elif len(size) < 2:
            self.size = size[0], size[0]
        else:
            self.size = size[0], size[1]  # Convention (width, height)
            self.match_min_size = False

    def __call__(self, img, mask):
        w, h = img.size
        out_w, out_h = self.size

        # if only one size was specified, match smaller side so specified size
        if self.match_min_size:
            if w < h:
                out_h = int(out_w * h / w)
            else:
                out_w = int(out_h * w / h)

        if (w <= h and w == out_w) or (h <= w and h == out_h):
            return img, mask

        img = img.resize((out_w, out_h), Image.BICUBIC)
        mask = mask.resize((out_w, out_h), Image.NEAREST)
        return img, mask


class RandomHorizontalFlip(object):
    def __init__(self, prob_flip=0.5):
        self.prob_flip = prob_flip

    def __call__(self, img, mask):
        if random.random() < self.prob_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomGaussianBlur(object):
    def __init__(self, prob_blur=0.5):
        self.prob_blur = prob_blur

    def __call__(self, img, mask):
        if random.random() < self.prob_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask


class RandomColorJitter(object):
    def __init__(self, s=0.25, prob_jitter=0.8):
        self.prob_jitter = prob_jitter
        self.color_jitter = ColorJitter(brightness=s, contrast=s, saturation=s, hue=s)

    def __call__(self, img, mask):
        if random.random() < self.prob_jitter:
            img = self.color_jitter(img)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, pad_coords=None, pad_val_img=0, pad_val_mask=255):
        if isinstance(size, int):
            width, height = size, size
        elif len(size) < 2:
            width, height = size[0], size[0]
        else:
            width, height = size[0], size[1]  # Convention (width, height)

        self.size = (width, height)
        self.pad_coords = pad_coords
        self.pad_val_img = pad_val_img
        self.pad_val_mask = pad_val_mask

    def __call__(self, img, mask):
        if mask is not None:
            if img.size != mask.size:
                if np.prod(img.size) > np.prod(mask.size):
                    new_size = img.size
                else:
                    new_size = mask.size
                img = img.resize(new_size, Image.BICUBIC)
                mask = mask.resize(new_size, Image.NEAREST)

        if self.pad_coords is not None:
            img = ImageOps.expand(img, border=self.pad_coords, fill=self.pad_val_img)
            mask = ImageOps.expand(mask, border=self.pad_coords, fill=self.pad_val_mask)

        w, h = img.size
        new_w, new_h = min(self.size[0], w), min(self.size[1], h)
        if w <= new_w and h <= new_h:
            return img, mask

        # get random crop coordinatesd
        x1 = random.randint(0, max(w - new_w, 0))
        y1 = random.randint(0, max(h - new_h, 0))
        crop_coords = (x1, y1, x1 + new_w, y1 + new_h)

        # do cropping
        img = img.crop(crop_coords)
        mask = mask.crop(crop_coords)
        return img, mask


class RandomScaleResize(object):
    def __init__(self, base_size, min_scale, max_scale):
        self.match_min_size = True
        if isinstance(base_size, int):
            new_width, new_height = base_size, base_size
        elif len(base_size) < 2:
            new_width, new_height = base_size[0], base_size[0]
        else:
            new_width, new_height = base_size[0], base_size[1]  # Convention (width, height)
            self.match_min_size = False

        self.base_size = (new_width, new_height)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, mask):
        w, h = img.size
        base_w, base_h = self.base_size

        # if only one size was specified, match smaller side so specified size
        if self.match_min_size:
            if w < h:
                base_h = int(base_w * h / w)
            else:
                base_w = int(base_h * w / h)

        # randomly scale the base_size by keeping the aspect ratio
        if (base_w / w) > (base_h / h):
            out_w = random.randint(int(base_w * self.min_scale), int(base_w * self.max_scale))
            out_h = int(h * out_w / w)
        else:
            out_h = random.randint(int(base_h * self.min_scale), int(base_h * self.max_scale))
            out_w = int(w * out_h / h)

        # resize image and mask to randomly scaled base_size
        img = img.resize((out_w, out_h), Image.BICUBIC)
        mask = mask.resize((out_w, out_h), Image.NEAREST)
        return img, mask


class Pad(object):
    def __init__(self, size, fill_img=0, fill_mask=255):
        if isinstance(size, int):
            width, height = size, size
        elif len(size) < 2:
            width, height = size[0], size[0]
        else:
            width, height = size[0], size[1]  # Convention (width, height)
        self.size = (width, height)

        if isinstance(fill_img, np.ndarray):
            fill_img = fill_img.astype(np.int32).tolist()
        self.fill_img = tuple([int(i) for i in fill_img[::-1]])  # img mean = BGR; image = RGB
        self.fill_mask = fill_mask

    def __call__(self, img, mask):
        w, h = img.size
        out_w, out_h = self.size
        # pad image and mask if they are smaller than the crop size
        if w < out_w or h < out_h:
            pad_total_h = max(out_h - h, 0)
            pad_total_w = max(out_w - w, 0)

            pad_left = random.randint(0, pad_total_w)
            pad_right = pad_total_w - pad_left
            pad_top = random.randint(0, pad_total_h)
            pad_bottom = pad_total_h - pad_top

            # if a mean RGB is subtracted in later processing step, it should be also used here for filling
            # This helps to keep the influence on the mean statistics of BN layers small
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_img)
            # ATTENTION: value used for filling should be some void label
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_mask)

        return img, mask
