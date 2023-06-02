import os
import logging

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from models.style_transfer import TransferNet
from datasets.data_loading import get_source_loader

logger = logging.getLogger(__name__)


class GTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                               batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        self.steps_adain = cfg.GTTA.STEPS_ADAIN
        self.use_style_transfer = cfg.GTTA.USE_STYLE_TRANSFER
        self.lam = cfg.GTTA.LAMBDA_MIXUP
        self.buffer_size = 100000
        self.counter = 0
        ckpt_dir = cfg.CKPT_DIR
        ckpt_path = cfg.CKPT_PATH

        self.avg_conf = torch.tensor(0.9).cuda()
        self.ignore_label = -1

        # Create style-transfer network
        if self.use_style_transfer:
            fname = os.path.join(ckpt_dir, "adain", f"decoder_{dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth" if dataset_name == "domainnet126" else f"decoder_{dataset_name}.pth")
            self.adain_model = TransferNet(ckpt_path_vgg=os.path.join(ckpt_dir, "adain", "vgg_normalized.pth"),
                                           ckpt_path_dec=fname,
                                           data_loader=self.src_loader,
                                           num_iters_pretrain=cfg.GTTA.PRETRAIN_STEPS_ADAIN).cuda()
            self.moments_list = [[torch.tensor([], device="cuda"), torch.tensor([], device="cuda")] for _ in range(2)]
            self.models = [self.model, self.adain_model]
        else:
            self.adain_model = None
            self.moments_list = None
            self.models = [self.model]

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        with torch.no_grad():
            outputs_test = self.model(imgs_test)

        if self.counter == 0:
            self.filtered_pseudos = self.create_pseudo_labels(outputs_test)
            if self.use_style_transfer:
                self.adain_model.train()
                self.extract_moments(imgs_test)

                # Train adain model
                for _ in range(self.steps_adain):
                    # sample source batch
                    try:
                        batch = next(self.src_loader_iter)
                    except StopIteration:
                        self.src_loader_iter = iter(self.src_loader)
                        batch = next(self.src_loader_iter)

                    # train on source data
                    imgs_src = batch[0].cuda()

                    self.adain_model.opt_adain_dec.zero_grad()
                    _, loss_content, loss_style = self.adain_model(imgs_src, moments_list=self.moments_list)
                    loss_adain = 1.0 * loss_content + 0.1 * loss_style
                    loss_adain.backward()
                    self.adain_model.opt_adain_dec.step()

        # Train classification model
        with torch.no_grad():
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # train on labeled source data
            imgs_src, labels_src = batch[0].cuda(), batch[1].cuda().long()

            if self.use_style_transfer:
                # Generate style transferred images from source images
                imgs_src, _, _ = self.adain_model(imgs_src, moments_list=self.moments_list)
            else:
                # Perform mixup
                batch_size = imgs_test.shape[0]
                imgs_src = imgs_src[:batch_size]
                labels_src = labels_src[:batch_size]
                outputs_src = self.model(imgs_src)
                _, ids = torch.max(torch.matmul(outputs_src.softmax(1), outputs_test.softmax(1).T), dim=1)
                imgs_src = self.mixup_data(imgs_src, imgs_test[ids], lam=self.lam)

        loss_source = F.cross_entropy(input=self.model(imgs_src), target=labels_src)
        loss_source.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        outputs_test = self.model(imgs_test)
        loss_target = F.cross_entropy(input=outputs_test, target=self.filtered_pseudos, ignore_index=-1)
        loss_target.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.counter += 1
        self.counter %= self.steps
        return outputs_test

    @torch.no_grad()
    def mixup_data(self, x_source, x_target, lam=0.25):
        mixed_x = lam * x_target + (1 - lam) * x_source
        return mixed_x

    @torch.no_grad()
    def create_pseudo_labels(self, outputs_test):
        # Create pseudo-labels
        confidences, pseudo_labels = torch.max(outputs_test.softmax(dim=1), dim=1)

        momentum = 0.9
        self.avg_conf = momentum * self.avg_conf + (1 - momentum) * confidences.mean()
        mask = torch.where(confidences < torch.sqrt(self.avg_conf))

        filtered_pseudos = pseudo_labels.clone()
        filtered_pseudos[mask] = self.ignore_label

        return filtered_pseudos

    @torch.no_grad()
    def extract_moments(self, x):
        # Extract image-wise moments from current test batch
        adain_moments = self.adain_model(x)

        # Save moments in a buffer list
        for i_adain_layer, (means, stds) in enumerate(adain_moments):  # Iterate through the adain layers
            self.moments_list[i_adain_layer][0] = torch.cat([self.moments_list[i_adain_layer][0], means], dim=0)
            self.moments_list[i_adain_layer][1] = torch.cat([self.moments_list[i_adain_layer][1], stds], dim=0)
            moments_size = len(self.moments_list[i_adain_layer][0])
            if moments_size > self.buffer_size:
                self.moments_list[i_adain_layer][0] = self.moments_list[i_adain_layer][0][moments_size - self.buffer_size:]
                self.moments_list[i_adain_layer][1] = self.moments_list[i_adain_layer][1][moments_size - self.buffer_size:]

    def reset(self):
        super().reset()
        self.moments_list = [[torch.tensor([], device="cuda"), torch.tensor([], device="cuda")] for _ in range(2)]

    def configure_model(self):
        """Configure model."""
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
