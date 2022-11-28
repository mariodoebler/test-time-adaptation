"""
Adapted from: https://github.com/Maggiking/AdaIN-Style-Transfer-PyTorch
"""

import os
import torch
import logging
import torch.nn as nn
import numpy as np
import torch.nn. functional as F

logger = logging.getLogger(__name__)

COUNT = 6


vggnet = nn.Sequential(
            # encode 1-1
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True), # relu 1-1
            # encode 2-1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True), # relu 2-1
            # encoder 3-1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True), # relu 3-1
            # encoder 4-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True), # relu 4-1
            # rest of vgg not used
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True)
)

# encoder for AdaIN model
encoder = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True)
)

# decoder for AdaIN model
decoder = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
    nn.Sigmoid()
)


class AdaIN(nn.Module):
    def __init__(self, num_classes, device, eps=1e-5, ignore_label=255):
        """
        The style means and standard deviations are drawn from a dictionary. This is more efficient as style images
        do not need to be encoded every time.
        :param num_classes: Number of classes
        :param device: Device to put tensors on
        :param eps: ensures no division with 0
        """
        super().__init__()
        self.eps = eps
        self.device = device
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def forward(self, x_content, y_content, moments_list, i_skip=0, pre_train=False):
        bs, ch, _, _ = x_content.shape

        # initialize mean and std with global moments
        mean_content = torch.zeros_like(x_content)
        mean_style = torch.zeros_like(x_content)
        std_content = torch.ones_like(x_content)
        std_style = torch.ones_like(x_content)

        # initialize some other variables
        means_list, stds_list = moments_list
        shape = (bs, self.num_classes, ch)  # (num_samples, num_classes, channels)
        style_means_1dim = torch.zeros(shape, dtype=torch.float32, device=self.device)  # will be used for loss calc
        style_stds_1dim = torch.zeros(shape, dtype=torch.float32, device=self.device)   # will be used for loss calc
        classes_batch = []

        for i in range(bs):
            sample = x_content[i]
            label = y_content[i]

            # get unique classes in label mask
            uniques, counts = torch.unique(label, return_counts=True)
            classes_cont = [c.item() for c, count in zip(uniques.cpu(), counts.cpu()) if c.item() != self.ignore_label and count.item() > COUNT]
            classes_used = []

            for class_nr in classes_cont:
                num_moments = means_list[i_skip][class_nr].shape[0]
                if num_moments > 0:
                    classes_used.append(class_nr)

                    # extract class-wise moments for content representation
                    mask = label == class_nr
                    feature_one_class = sample[:, mask]
                    mean_c = torch.mean(feature_one_class, dim=1).reshape(1, -1, 1, 1)
                    std_c = torch.std(feature_one_class, dim=1).reshape(1, -1, 1, 1) + self.eps
                    mean_content[i:i + 1] = torch.where(mask, mean_c, mean_content[i].unsqueeze(0))
                    std_content[i:i + 1] = torch.where(mask, std_c, std_content[i].unsqueeze(0))

                    # get style mean and std of a randomly picked sample number
                    rand_sample_nr = np.random.randint(num_moments) if i >= 1 or pre_train else -1
                    mean_s = means_list[i_skip][class_nr][rand_sample_nr].to(self.device)
                    std_s = stds_list[i_skip][class_nr][rand_sample_nr].to(self.device)

                    style_means_1dim[i, class_nr, :] += mean_s
                    style_stds_1dim[i, class_nr, :] += std_s

                    mean_s = mean_s.reshape(1, -1, 1, 1)
                    std_s = std_s.reshape(1, -1, 1, 1)
                    mean_style[i:i + 1] = torch.where(mask, mean_s, mean_style[i].unsqueeze(0))
                    std_style[i:i + 1] = torch.where(mask, std_s, std_style[i].unsqueeze(0))

            # track used classes for style transfer
            classes_batch.append(classes_used)

        out = (x_content - mean_content) / std_content * std_style + mean_style
        return out, style_means_1dim, style_stds_1dim, classes_batch


class MomentExtraction(nn.Module):
    def __init__(self, num_classes, device, eps=1e-5):
        """
        Calculate the class-wise means and standard deviations
        :param num_classes: Number of classes
        :param device: Device to put tensors on
        :param eps: Ensures no division with 0
        """
        super().__init__()
        self.eps = eps
        self.device = device
        self.num_classes = num_classes

    def forward(self, x, y, classes=None):
        shape = (x.shape[0], self.num_classes, x.shape[1])  # (num_samples, num_classes, channels)
        means = torch.zeros(shape, dtype=torch.float32, device=self.device)
        stds = torch.zeros(shape, dtype=torch.float32, device=self.device)
        classes_batch = [] if classes is None else classes

        for i in range(x.shape[0]):  # go through all the samples in one batch
            sample, label = x[i], y[i]

            if classes is None:
                # get all classes contained in the image
                uniques, counts = torch.unique(label, return_counts=True)
                classes_batch.append([c.item() for c, count in zip(uniques.cpu(), counts.cpu()) if c.item() != 255 and count.item() > COUNT])

            cl = classes_batch[i]
            for class_nr in cl:
                mask = label == class_nr
                feature_one_class = sample[:, mask]
                means[i, class_nr, :] += torch.mean(feature_one_class, dim=1)
                stds[i, class_nr, :] += torch.std(feature_one_class, dim=1) + self.eps

        return means, stds, classes_batch


class TransferNet(nn.Module):
    def __init__(self, ckpt_path_vgg, ckpt_path_dec, src_loader, device, num_iters_pretrain=20000, num_classes=14):
        """
        Style transfer network
        :param args: arguments
        :param device: Device to put tensors on
        """
        super().__init__()
        self.mse_criterion = nn.MSELoss()
        self.src_loader = src_loader
        # self.src_loader_iter = iter(src_loader)
        self.num_classes = num_classes

        # get pre-trained vgg19 model
        vgg_model = torch.load(ckpt_path_vgg)
        vggnet.load_state_dict(vgg_model)

        # create and freeze the encoder
        self.encoder = encoder.to(device)
        self.encoder.load_state_dict(vggnet[:21].state_dict())
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # create trainable decoder and adain layer with spatial control
        self.decoder = decoder.to(device)

        # create trainable decoder and adain layer with spatial control
        self.adain = AdaIN(num_classes, device=device)
        self.get_moments = MomentExtraction(num_classes, device=device)

        # pre-train decoder if no checkpoint exists
        if not os.path.isfile(ckpt_path_dec):
            logger.info(f"Start pre-training the style transfer model...")
            self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
            self.pretrain_adain(final_ckpt_path=ckpt_path_dec, num_iters=num_iters_pretrain)

        # load checkpoint
        checkpoint = torch.load(ckpt_path_dec, map_location=device)
        self.decoder.load_state_dict(checkpoint['decoder'])
        logger.info(f"Successfully loaded AdaIN checkpoint: {ckpt_path_dec}")
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

    def pretrain_adain(self, final_ckpt_path, num_iters=20000):
        moments_list = self.get_moments_list()

        # start the actual training
        self.train()
        avg_loss = 0
        avg_loss_content = 0
        avg_loss_style = 0

        for i in range(1, num_iters + 1):
            try:
                imgs_src, labels_src, ids_src = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                imgs_src, labels_src, ids_src = next(self.src_loader_iter)

            self.optimizer_dec.zero_grad()
            gen_imgs, loss_content, loss_style = self.adain_model(
                input=[imgs_src.to(self.device), labels_src.to(self.device).long()], moments_list=moments_list, pre_train=True)

            loss = loss_content + 0.1 * loss_style
            loss.backward()
            self.optimizer_dec.step()

            # log current losses
            avg_loss += loss.item()
            avg_loss_content += loss_content.item()
            avg_loss_style += loss_style.item()

            if i % 500 == 0:
                logger.info(f"[{i}/{num_iters}] loss: {avg_loss / 500:.4f}, "
                            f"content: {avg_loss_content / 500:.4f}, style: {avg_loss_style / 500:.4f}")
                avg_loss = 0
                avg_loss_content = 0
                avg_loss_style = 0

        ckpt_dict = {'decoder': self.decoder.state_dict()}
        torch.save(ckpt_dict, final_ckpt_path)
        logger.info(f"Saved pre-trained AdaIN model to: {final_ckpt_path}")

    def get_moments_list(self):
        dataset_means = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(2)]
        dataset_stds = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(2)]

        # extract source moments using original source images
        self.src_loader.dataset_name.is_training = False

        with torch.no_grad():
            for imgs_src, labels_src, ids_src in self.src_loader:
                # extract the class-wise moments
                out_adain = self.adain_model(input=[imgs_src.to(self.device), labels_src.to(self.device).long()])

                for i_adain_layer, (means, stds, classes) in enumerate(out_adain):  # iterate through the adain layers
                    for i_sample in range(means.shape[0]):  # iterate through all samples of one batch
                        for class_nr in classes[i_sample]:  # iterate through all classes contained in one sample
                            # only add moment, if the class is present in both adain layers
                            if class_nr in out_adain[-1][-1][i_sample]:
                                dataset_means[i_adain_layer][class_nr] = torch.cat(
                                    [dataset_means[i_adain_layer][class_nr], means[i_sample, class_nr, :].unsqueeze(0).cpu()], dim=0)
                                dataset_stds[i_adain_layer][class_nr] = torch.cat(
                                    [dataset_stds[i_adain_layer][class_nr], stds[i_sample, class_nr, :].unsqueeze(0).cpu()], dim=0)
        logger.info("Successfully extracted all source moments for pre-training!")

        self.src_loader.dataset_name.is_training = True
        return [dataset_means, dataset_stds]

    def forward(self, input, moments_list=None, pre_train=False):
        # propagate the input image through the encoder
        imgs, labels = input
        fm11_enc = self.encoder[:5](imgs)
        out_encoder = self.encoder[5:](fm11_enc)

        # resize the label masks to match the size of the encoded images
        labels_enc = F.interpolate(labels.float().unsqueeze(1), size=out_encoder.shape[2:], mode='nearest').squeeze(1).long()

        if moments_list is not None:  # perform style transfer with a list containing the moments
            # perform style transfer for output encoder: out = sigma_style * [(x - mu_cont) / sigma_cont] + mu_style
            out_encoder, means_style_enc, stds_style_enc, classes_enc = self.adain(out_encoder, labels_enc,
                                                                                   moments_list=moments_list,
                                                                                   i_skip=1, pre_train=pre_train)

            # partially decode the output of the encoder
            fm11_dec = self.decoder[:17](out_encoder)

            # perform style transfer for feature map11: out = sigma_style * [(x - mu_cont) / sigma_cont] + mu_style
            fm11_enc, means_style_11, stds_style_11, classes_11 = self.adain(fm11_enc, labels.long(),
                                                                             moments_list=moments_list,
                                                                             i_skip=0, pre_train=pre_train)
            # skip connection with former adain transformation
            fm11_dec = torch.add(fm11_dec, fm11_enc)
            gen_img = self.decoder[17:](fm11_dec)
        else:
            # extract the moments of the encoded feature maps of the input
            means_fm11, stds_fm11, classes_11 = self.get_moments(fm11_enc, labels.long())
            means_enc, stds_enc, classes_enc = self.get_moments(out_encoder, labels_enc)
            return [means_fm11, stds_fm11, classes_11], [means_enc, stds_enc, classes_enc]

        if self.training:
            # encode style transferred images again
            fm11_gen = self.encoder[:5](gen_img)
            encode_gen = self.encoder[5:](fm11_gen)

            means_gen_11, stds_gen_11, _ = self.get_moments(fm11_gen, labels, classes_11)
            means_gen_enc, stds_gen_enc, _ = self.get_moments(encode_gen, labels_enc, classes_enc)

            # calculate  content and style loss
            loss_content = self.mse_criterion(encode_gen, out_encoder)
            loss_style = self.mse_criterion(means_gen_11, means_style_11) + self.mse_criterion(means_gen_enc, means_style_enc) + \
                         self.mse_criterion(stds_gen_11, stds_style_11) + self.mse_criterion(stds_gen_enc, stds_style_enc)

            return gen_img, loss_content, loss_style

        else:
            return gen_img
