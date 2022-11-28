
import os
import wandb
import logging

import torch
import torch.optim as optim

from conf import cfg, load_cfg_fom_args
from utils.arch_utils import load_model
from utils.eval_utils import evaluate_sequence
from datasets.carla_dataset import create_carla_loader, IMG_MEAN
from augmentations.transforms_source import get_src_transform
from models.style_transfer_cace import TransferNet

from methods.bn import AlphaBatchNorm
from methods.norm_ema import NormEMA
from methods.tent import Tent
from methods.memo import MEMO
from methods.cotta import CoTTA
from methods.gtta import GTTA
from methods.asm import ASM
from methods.sm_ppm import SMPPM

os.environ["WANDB_MODE"] = "offline"
logger = logging.getLogger(__name__)


def main(description):
    load_cfg_fom_args(description)

    # check available devices
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info('\nModel will run on {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        device = torch.device('cpu')
        logger.info('\nModel will run on CPU')

    # setup wandb logging
    wandb.init(project="GradualDomainAdaptation",
               name=cfg.EXP_NAME,
               config=cfg,
               resume="allow")

    # Setup segmentation model
    base_model = load_model(method=cfg.MODEL.ADAPTATION,
                            ckpt_path=cfg.CKPT_PATH_SEG,
                            device=device,
                            img_mean=IMG_MEAN,
                            imagenet_init=cfg.MODEL.IMAGENET_INIT,
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            model_name=cfg.MODEL.NAME)

    # setup test data loader
    test_loader = create_carla_loader(data_dir=cfg.DATA_DIR,
                                      list_path=cfg.LIST_NAME_TEST,
                                      ignore_label=cfg.OPTIM.IGNORE_LABEL,
                                      test_size=cfg.TEST.IMG_SIZE,
                                      batch_size=cfg.TEST.BATCH_SIZE,
                                      workers=1,
                                      is_training=False)

    if cfg.MODEL.ADAPTATION == "source":        # BN--0
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_test":   # BN--1
        model = setup_test_norm(base_model, device)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model = setup_alpha_norm(base_model, device)
    elif cfg.MODEL.ADAPTATION == "norm_ema":    # BN--EMA
        model = setup_ema_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        model = setup_tent(base_model)
    elif cfg.MODEL.ADAPTATION == "memo":
        model = setup_memo(base_model, device)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model = setup_cotta(base_model)
    elif cfg.MODEL.ADAPTATION == "asm":
        model = setup_asm(base_model, device)
    elif cfg.MODEL.ADAPTATION == "sm_ppm":
        model = setup_smppm(base_model, device)
    elif cfg.MODEL.ADAPTATION == "gtta":
        model = setup_gtta(base_model, device)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    preds_dir_path = os.path.join(cfg.SAVE_DIR, "predictions")
    if cfg.SAVE_PREDICTIONS:
        os.makedirs(preds_dir_path, exist_ok=True)

    # start adapting the model during test time
    miou = evaluate_sequence(model=model,
                             data_loader=test_loader,
                             device=device,
                             num_classes=cfg.MODEL.NUM_CLASSES,
                             save_preds=cfg.SAVE_PREDICTIONS,
                             preds_dir_path=preds_dir_path)

    logger.info(f"Final mIoU for complete sequence: {cfg.LIST_NAME_TEST} \t mIoU = {miou}")


def setup_source(model):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model


def setup_test_norm(model, device):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = AlphaBatchNorm.adapt_model(model, alpha=1.0).to(device)  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model


def setup_alpha_norm(model, device):
    """Set up BN--0.1 (test-time normalization adaptation with source prior).
    Normalize features by combining the source moving statistics and the test batch statistics.
    """
    norm_model = AlphaBatchNorm.adapt_model(model, alpha=cfg.BN.ALPHA).to(device)  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model


def setup_ema_norm(model):
    """Set up BN--EMA (test-time normalization adaptation using an exponential moving average).
    """
    model.train()
    norm_model = NormEMA(model, episodic=cfg.MODEL.EPISODIC)
    return norm_model


def setup_tent(model):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = Tent(model, optimizer,
                      crop_size=cfg.SOURCE.CROP_SIZE,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC)
    return tent_model


def setup_memo(model, device):
    model = AlphaBatchNorm.adapt_model(model, alpha=cfg.BN.ALPHA).to(device)
    params_feat, params_head = MEMO.collect_params(model)
    optimizer = setup_optimizer(params_feat, params_head, cfg.CKPT_PATH_SEG)
    memo_model = MEMO(model, optimizer,
                      crop_size=cfg.SOURCE.CROP_SIZE,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      n_augmentations=cfg.TEST.N_AUGMENTATIONS)
    return memo_model


def setup_cotta(model):
    model.train()
    params_feat, params_head = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params_feat, params_head, cfg.CKPT_PATH_SEG)
    cotta_model = CoTTA(model, optimizer,
                        crop_size=cfg.SOURCE.CROP_SIZE,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        n_augmentations=cfg.TEST.N_AUGMENTATIONS,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP)
    return cotta_model


def setup_asm(model, device):
    params_feat, params_head = ASM.collect_params(model)
    optimizer = setup_optimizer(params_feat, params_head, cfg.CKPT_PATH_SEG)
    asm_model = ASM(model, optimizer,
                    crop_size=cfg.SOURCE.CROP_SIZE,
                    steps=cfg.OPTIM.STEPS,
                    episodic=cfg.MODEL.EPISODIC,
                    src_loader=setup_src_loader(cfg, IMG_MEAN),
                    ckpt_dir=cfg.CKPT_DIR,
                    device=device,
                    img_mean=IMG_MEAN,
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    ignore_label=cfg.OPTIM.IGNORE_LABEL)
    return asm_model


def setup_smppm(model, device):
    params_feat, params_head = SMPPM.collect_params(model)
    optimizer = setup_optimizer(params_feat, params_head, cfg.CKPT_PATH_SEG, cfg.MODEL.ADAPTATION)
    smppm_model = SMPPM(model, optimizer,
                        crop_size=cfg.SOURCE.CROP_SIZE,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        src_loader=setup_src_loader(cfg, IMG_MEAN),
                        device=device,
                        ignore_label=cfg.OPTIM.IGNORE_LABEL)
    return smppm_model


def setup_gtta(model, device):
    params_feat, params_head = GTTA.collect_params(model)
    optimizer = setup_optimizer(params_feat, params_head, cfg.CKPT_PATH_SEG)

    adain_src_loader = setup_src_loader(cfg, IMG_MEAN, batch_size=4, min_scale=1.0, crop_size=(512, 512))
    adain_model = TransferNet(ckpt_path_vgg=os.path.join(cfg.CKPT_DIR, "vgg_normalized.pth"),
                              ckpt_path_dec=cfg.CKPT_PATH_ADAIN_DEC,
                              src_loader=adain_src_loader,
                              device=device,
                              num_iters_pretrain=cfg.GTTA.PRETRAIN_STEPS_ADAIN,
                              num_classes=cfg.MODEL.NUM_CLASSES)

    src_loader = setup_src_loader(cfg, IMG_MEAN)
    gtta_model = GTTA(model, optimizer,
                      crop_size=cfg.SOURCE.CROP_SIZE,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      adain_model=adain_model,
                      src_loader=src_loader,
                      adain_loader=adain_src_loader,
                      steps_adain=cfg.GTTA.STEPS_ADAIN,
                      device=device,
                      save_dir=cfg.SAVE_DIR,
                      lambda_ce_trg=cfg.GTTA.LAMBDA_CE_TRG,
                      num_classes=cfg.MODEL.NUM_CLASSES,
                      ignore_label=cfg.OPTIM.IGNORE_LABEL,
                      conf_thresh=cfg.GTTA.CONF_THRESH,
                      class_weighting=cfg.GTTA.USE_CLASS_WEIGHTING,
                      style_transfer=cfg.GTTA.USE_STYLE_TRANSFER)
    return gtta_model


def setup_optimizer(params, params_head=None, ckpt_path=None, method=None):
    if cfg.OPTIM.METHOD == 'SGD':
        optimizer = optim.SGD(params,
                              lr=cfg.OPTIM.LR,
                              momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WD,
                              nesterov=cfg.OPTIM.NESTEROV)
        if params_head is not None:
            optimizer.add_param_group({'params': params_head, 'lr': cfg.OPTIM.SCALE_LR_SEGHEAD * cfg.OPTIM.LR})
    else:
        raise NotImplementedError

    # restore optimizer if a checkpoint is provided
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if method == "sm_ppm":  # sm-ppm uses the same constant lr for all parameters
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.OPTIM.LR
        logger.info("Successfully restored optimizer")
    return optimizer


def setup_src_loader(cfg, img_mean, batch_size=None, min_scale=None, crop_size=None):
    transform_train = get_src_transform(cfg, img_mean, min_scale=min_scale, crop_size=crop_size)
    src_loader = create_carla_loader(data_dir=cfg.DATA_DIR,
                                     list_path=cfg.LIST_NAME_SRC,
                                     ignore_label=cfg.OPTIM.IGNORE_LABEL,
                                     test_size=cfg.TEST.IMG_SIZE,
                                     batch_size=cfg.SOURCE.BATCH_SIZE if batch_size is None else batch_size,
                                     percentage=cfg.SOURCE.PERCENTAGE,
                                     workers=cfg.OPTIM.WORKERS,
                                     transform_train=transform_train,
                                     is_training=True)
    return src_loader


if __name__ == '__main__':
    main("Evaluation.")
