"""
Builds upon: https://github.com/W-zx-Y/SM-PPM
Paper: https://arxiv.org/pdf/2112.04665.pdf
"""

import logging
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange
from methods.base import TTAMethod
from torchvision.transforms import Compose, RandomCrop

logger = logging.getLogger(__name__)


class SMPPM(TTAMethod):
    def __init__(self, model, optimizer, crop_size, steps, episodic, src_loader, device, ignore_label=255):
        super().__init__(model, optimizer, crop_size, steps, episodic)
        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.ignore_label = ignore_label
        self.device = device

        # initialize some other variables
        self.seg_loss = UncCELoss(ignore=ignore_label).to(self.device)
        self.interp_fea = torch.nn.Upsample(size=(128, 256), mode='bilinear', align_corners=True)
        self.rand_crop = Compose([RandomCrop((960, 1920))])  # crop image into cityscapes ratio

    def forward(self, x):
        if self.episodic:
            self.reset()

        self.model.train()
        for _ in range(self.steps):
            self.forward_and_adapt(x)

        self.model.eval()
        pred, _ = self.model([x, None])
        return pred

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()

        with torch.no_grad():
            _, t_feas = self.model([self.rand_crop(x), None])
            _, _, h, w = t_feas[2].shape
            p = 32
            ref_patch4 = rearrange(self.interp_fea(t_feas[4]), 'b c (h p1) (w p2) -> (h w) b c (p1 p2)', p1=p, p2=p)
            prototypes4 = [torch.mean(ref_patch4[i], dim=2) for i in range(ref_patch4.shape[0])]

        # sample source batch
        try:
            imgs_src, labels_src, files_src = next(self.src_loader_iter)
        except StopIteration:
            self.src_loader_iter = iter(self.src_loader)
            imgs_src, labels_src, files_src = next(self.src_loader_iter)

        pred, s_feas = self.model([imgs_src.to(self.device), t_feas])
        pred = F.interpolate(pred, size=imgs_src.shape[2:], mode='bilinear', align_corners=True)
        entropy_s = torch.from_numpy(compute_entropy(pred)).to(self.device)

        # achieve the similarity using the feature from layer4
        conf = similarity(s_feas[4], prototypes4)
        conf = torch.max(conf, dim=1)[0]
        conf = conf.unsqueeze(1)
        conf = F.interpolate(conf, size=imgs_src.shape[2:], mode='bilinear', align_corners=True)

        loss = self.seg_loss(pred, labels_src.long().to(self.device), conf * (1 - entropy_s))
        loss.backward()
        self.optimizer.step()


class UncCELoss(torch.nn.Module):
    def __init__(self, num_classes=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(UncCELoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight
        self.raw = False
        if (num_classes < 19):
            self.raw = True

    def forward(self, input, target, conf, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)
        conf = conf.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            conf = conf[valid]

        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))

        probs = F.log_softmax(input, dim=1)
        probs = (probs * target_onehot)
        probs = torch.sum(probs, dim=1)
        probs = conf * probs + probs
        batch_loss = -probs
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def compute_entropy(pred):
    output_sm = F.softmax(pred, dim=1).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm + 1e-30)), axis=2, keepdims=False)
    output_ent = output_ent/np.log2(19)
    return output_ent


def similarity(s_fea, prototypes):
    conf = [F.cosine_similarity(s_fea, prototype[..., None, None]) for prototype in prototypes]
    conf = torch.stack(conf, dim=1)
    return conf


def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

