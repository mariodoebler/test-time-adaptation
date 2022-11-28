"""
Builds upon: https://github.com/RoyalVane/ASM
Corresponding paper: https://proceedings.neurips.cc/paper/2020/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.rain_asm import encoder, decoder, fc_encoder, fc_decoder
from methods.base import TTAMethod
from conf import map_asm_rain_paths

logger = logging.getLogger(__name__)


class ASM(TTAMethod):
    def __init__(self, model, optimizer, crop_size, steps, episodic, src_loader, ckpt_dir, device, img_mean, num_classes=14, ignore_label=255):
        super().__init__(model, optimizer, crop_size, steps, episodic)

        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.device = device
        self.img_mean = torch.FloatTensor(img_mean).to(self.device).view(1, 3, 1, 1)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        self.loss_norm = torch.nn.MSELoss()
        self.input_shape_src = tuple(crop_size[::-1])
        self.interp_source = torch.nn.Upsample(size=self.input_shape_src, mode='bilinear', align_corners=True)

        ###################### prepare style transfer network ####################
        self.vgg_encoder = encoder
        self.vgg_decoder = decoder
        self.style_encoder = fc_encoder
        self.style_decoder = fc_decoder

        self.vgg_encoder.eval()
        self.style_encoder.eval()
        self.vgg_decoder.eval()
        self.style_decoder.eval()

        self.vgg_encoder.load_state_dict(torch.load(map_asm_rain_paths(ckpt_dir, model_name='vgg_enc_path')))
        self.vgg_encoder = torch.nn.Sequential(*list(self.vgg_encoder.children())[:31])
        self.vgg_decoder.load_state_dict(torch.load(map_asm_rain_paths(ckpt_dir, model_name='vgg_dec_path')))
        self.style_encoder.load_state_dict(torch.load(map_asm_rain_paths(ckpt_dir, model_name='style_enc_path')))
        self.style_decoder.load_state_dict(torch.load(map_asm_rain_paths(ckpt_dir, model_name='style_dec_path')))

        self.vgg_encoder.to(device)
        self.vgg_decoder.to(device)
        self.style_encoder.to(device)
        self.style_decoder.to(device)

        for param in self.vgg_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.episodic:
            self.reset()

        # Prepare test image for style transfer with source data
        x_new = F.interpolate(x.clone(), size=self.input_shape_src, mode='bilinear', align_corners=True)
        x_new = Variable(x_new).to(self.device)
        x_new.requires_grad = False

        self.model.train()
        for _ in range(self.steps):
            _ = self.forward_and_adapt(x_new)

        self.model.eval()
        output, _ = self.model(x)
        return output

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, img_test):
        sampling = None

        # Train with Source
        # sample source batch
        try:
            imgs_src, labels_src, files_src = next(self.src_loader_iter)
        except StopIteration:
            self.src_loader_iter = iter(self.src_loader)
            imgs_src, labels_src, files_src = next(self.src_loader_iter)

        imgs_src = Variable(imgs_src).to(self.device)
        imgs_src.requires_grad = False

        for i in range(2):
            self.optimizer.zero_grad()
            imgs_src_style, sampling = style_transfer(self.vgg_encoder, self.vgg_decoder,
                                                      self.style_encoder, self.style_decoder,
                                                      imgs_src, img_test, sampling)

            # note that the image normalization is happening in the segmentation model
            imgs_src_style = self.interp_source(imgs_src_style)
            pred, pred_norm = self.model(torch.cat([imgs_src_style, imgs_src], dim=0))
            pred = self.interp_source(pred)

            # Segmentation Loss
            loss_1 = loss_calc(pred, torch.cat([labels_src, labels_src], dim=0), device=self.device, num_classes=self.num_classes)
            loss_2 = self.loss_norm(pred_norm[39616:], torch.zeros(pred_norm[39616:].size(), device=self.device))

            sampling.retain_grad()
            loss = loss_1 + 2e-4 * loss_2
            loss.backward(retain_graph=True)

            sampling = sampling + (20.0 / loss.item()) * sampling.grad.data
            self.optimizer.step()


class CrossEntropy2d(nn.Module):

    def __init__(self, class_num=19, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax(dim = 0)
        predict = predict.transpose(0, 1).contiguous()
        P = sm(predict)
        P = torch.clamp(P, min = 1e-9, max = 1- (1e-9))

        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = P[target_mask.view(1, N, H, W).repeat(C, 1, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim=0, index=target.view(1, -1))
        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:

            loss = batch_loss.sum()
        return loss


def loss_calc(pred, label, device, num_classes):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d(num_classes).to(device)
    return criterion(pred, label)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_feat_mean_std(input, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = input.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
    return torch.cat([feat_mean, feat_std], dim = 1)


def adaptive_instance_normalization_with_noise(content_feat, style_feat):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    N, C = size[:2]
    style_mean = style_feat[:, :512].view(N, C, 1, 1)
    style_std = style_feat[:, 512:].view(N, C, 1, 1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_transfer(encoder, decoder, fc_encoder, fc_decoder, content, style, sampling = None):
    with torch.no_grad():
        content_feat = encoder(content)
        style_feat = encoder(style)
    style_feat_mean_std = calc_feat_mean_std(style_feat)
    intermediate = fc_encoder(style_feat_mean_std)
    intermediate_mean = intermediate[:, :512]
    intermediate_std = intermediate[:, 512:]
    noise = torch.randn_like(intermediate_mean)
    if sampling is None:
        sampling = intermediate_mean + noise * intermediate_std #N, 512
    sampling.require_grad = True
    style_feat_mean_std_recons = fc_decoder(sampling) #N, 1024
    feat = adaptive_instance_normalization_with_noise(content_feat, style_feat_mean_std_recons)

    return decoder(feat), sampling
