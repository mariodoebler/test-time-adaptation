"""
Adapted from: https://github.com/Maggiking/AdaIN-Style-Transfer-PyTorch
"""

import os
import logging

import torch
import torch.nn as nn
import torch.jit


logger = logging.getLogger(__name__)

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
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x_content, moments_list, pretrain=False):
        means_style, stds_style = moments_list[0], moments_list[1]

        # extract the means and standard deviations of the content images
        means_content = torch.mean(x_content, dim=[2, 3], keepdim=True)
        stds_content = torch.std(x_content, dim=[2, 3], keepdim=True) + self.eps

        # get moments from moment_list
        rand_sample_nr = torch.randint(means_style.shape[0], size=x_content.shape[:1]) if pretrain else -1 * torch.arange(1, x_content.shape[0] + 1)
        means_style = means_style[rand_sample_nr].unsqueeze(-1).unsqueeze(-1)
        stds_style = stds_style[rand_sample_nr].unsqueeze(-1).unsqueeze(-1) + self.eps

        adain_out = (x_content - means_content) / stds_content * stds_style + means_style

        return adain_out, means_style.squeeze(), stds_style.squeeze()


class TransferNet(nn.Module):
    def __init__(self, ckpt_path_vgg, ckpt_path_dec, data_loader, num_iters_pretrain=20000):
        """
        Style transfer network
        :param vgg_model: Path to ImageNet pre-trained vgg19 model
        """
        super().__init__()
        self.mse_criterion = nn.MSELoss()
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)

        # get pre-trained vgg19 model
        vgg_model = torch.load(ckpt_path_vgg)
        vggnet.load_state_dict(vgg_model)

        # create and freeze the encoder
        self.encoder = encoder.cuda()
        self.encoder.load_state_dict(vggnet[:21].state_dict())
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # create trainable decoder and adain layer
        self.decoder = decoder.cuda()
        self.adain = AdaIN()

        # pre-train decoder if no checkpoint exists
        if not os.path.isfile(ckpt_path_dec):
            logger.info(f"Start pre-training the style transfer model...")
            self.opt_adain_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
            self.pretrain_adain(final_ckpt_path=ckpt_path_dec, num_iters=num_iters_pretrain)

        # load checkpoint
        checkpoint = torch.load(ckpt_path_dec, map_location="cuda")
        self.decoder.load_state_dict(checkpoint['decoder'])
        logger.info(f"Successfully loaded AdaIN checkpoint: {ckpt_path_dec}")
        self.opt_adain_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

    def pretrain_adain(self, final_ckpt_path, num_iters=20000):
        # initialize buffer lists
        moments_list = [[torch.tensor([], device="cuda"), torch.tensor([], device="cuda")] for _ in range(2)]
        n_samples = 0
        # extract all source moments
        with torch.no_grad():
            for images, labels in self.data_loader:
                n_samples += images.shape[0]
                # extract the class-wise moments
                out_adain = self.forward(images=images.cuda())

                # save moments in a buffer list
                for i_adain_layer, (means, stds) in enumerate(out_adain):  # iterate through the adain layers
                    moments_list[i_adain_layer][0] = torch.cat([moments_list[i_adain_layer][0], means], dim=0)
                    moments_list[i_adain_layer][1] = torch.cat([moments_list[i_adain_layer][1], stds], dim=0)

                if n_samples >= 100000:
                    break

        # start the actual training
        self.train()
        avg_loss = 0
        avg_loss_content = 0
        avg_loss_style = 0

        for i in range(1, num_iters + 1):
            try:
                images, labels_src = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                images, labels = next(self.data_loader_iter)

            # extract the class-wise moments
            self.opt_adain_dec.zero_grad()
            gen_img, loss_content, loss_style = self.forward(images=images.cuda(),
                                                             moments_list=moments_list,
                                                             pretrain=True)

            loss = loss_content + 0.1 * loss_style
            loss.backward()
            self.opt_adain_dec.step()

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

    def _calculate_moments(self, x):
        means = torch.mean(x, dim=[2, 3])
        stds = torch.std(x, dim=[2, 3])
        return means, stds

    def forward(self, images, moments_list=None, pretrain=False):
        # propagate the content image through the encoder
        fm11_enc = self.encoder[:5](images)
        out_encoder = self.encoder[5:](fm11_enc)

        if moments_list is None:
            # extract the moments of the encoded feature maps of the input
            means_fm11, stds_fm11 = self._calculate_moments(fm11_enc)
            means_enc, stds_enc = self._calculate_moments(out_encoder)
            return [means_fm11, stds_fm11], [means_enc, stds_enc]

        else:  # perform style transfer with a list containing the moments
            # perform style transfer for output encoder: out = sigma_style * [(x - mu_cont) / sigma_cont] + mu_style
            out_encoder, means_style_enc, stds_style_enc = self.adain(out_encoder, moments_list=moments_list[1], pretrain=pretrain)

            # partially decode the output of the encoder
            fm11_dec = self.decoder[:17](out_encoder)

            # perform style transfer for feature map11: out = sigma_style * [(x - mu_cont) / sigma_cont] + mu_style
            fm11_enc, means_style_11, stds_style_11 = self.adain(fm11_enc, moments_list=moments_list[0], pretrain=pretrain)
            # skip connection with former adain transformation
            fm11_dec = torch.add(fm11_dec, fm11_enc)
            gen_img = self.decoder[17:](fm11_dec)

        if self.training:
            # encode style transferred images again
            fm11_gen = self.encoder[:5](gen_img)
            encode_gen = self.encoder[5:](fm11_gen)

            means_gen_11, stds_gen_11 = self._calculate_moments(fm11_gen)
            means_gen_enc, stds_gen_enc = self._calculate_moments(encode_gen)

            # calculate  content and style loss
            loss_content = self.mse_criterion(encode_gen, out_encoder)
            loss_style = self.mse_criterion(means_gen_11, means_style_11) + self.mse_criterion(means_gen_enc, means_style_enc) + \
                         self.mse_criterion(stds_gen_11, stds_style_11) + self.mse_criterion(stds_gen_enc, stds_style_enc)

            return gen_img, loss_content, loss_style

        else:
            return gen_img
