from augmentations import augmentations as augs


def source_transform(random_crop=False,
                     random_scale_crop=True,
                     prob_flip=0.5,
                     prob_blur=0.5,
                     prob_jitter=0.8,
                     jitter_val=0.25,
                     min_scale=0.75,
                     max_scale=2.0,
                     base_size=512,
                     crop_size=(1024, 512),
                     ignore_label=255,
                     img_mean=None):

    # setup data processing pipeline
    transformations = []
    if random_scale_crop and min_scale != max_scale:
        transformations.append(augs.RandomScaleResize(base_size, min_scale=min_scale, max_scale=max_scale))
    else:
        transformations.append(augs.Resize(base_size))

    if random_crop or random_scale_crop:
        transformations.append(augs.RandomCrop(crop_size))
    if prob_blur > 0.:
        transformations.append(augs.RandomGaussianBlur(prob_blur))
    if prob_jitter > 0.:
        transformations.append(augs.RandomColorJitter(prob_jitter=prob_jitter, s=jitter_val))
    if prob_flip > 0.:
        transformations.append(augs.RandomHorizontalFlip(prob_flip=prob_flip))

    transformations.append(augs.Pad(crop_size, fill_img=img_mean, fill_mask=ignore_label))
    return augs.Compose(transformations)


def get_src_transform(cfg, img_mean, min_scale=None, crop_size=None):
    return source_transform(random_crop=cfg.SOURCE.RANDOM_CROP,
                            random_scale_crop=cfg.SOURCE.RANDOM_SCALE_CROP,
                            prob_flip=cfg.SOURCE.PROB_FLIP,
                            prob_blur=cfg.SOURCE.PROB_BLUR,
                            prob_jitter=cfg.SOURCE.PROB_JITTER,
                            jitter_val=cfg.SOURCE.JITTER_VAL,
                            min_scale=cfg.SOURCE.MIN_SCALE if min_scale is None else min_scale,
                            max_scale=cfg.SOURCE.MAX_SCALE,
                            base_size=cfg.SOURCE.BASE_SIZE,
                            crop_size=cfg.SOURCE.CROP_SIZE if crop_size is None else crop_size,
                            ignore_label=cfg.OPTIM.IGNORE_LABEL,
                            img_mean=img_mean)
