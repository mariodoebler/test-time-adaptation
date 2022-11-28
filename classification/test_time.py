
import logging
import math
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils import get_accuracy
from models.model import get_model
from conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
from datasets.data_loading import get_source_loader, get_test_loader

from methods.bn import AlphaBatchNorm
from methods.tent import Tent
from methods.cotta import CoTTA
from methods.gtta import GTTA
from methods.adacontrast import AdaContrast
from methods.rmt import RMT
from methods.norm import Norm


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.SETTING in ["continual", "reset_each_shift", "non_stationary", "correlated", "non_stationary_correlated"]

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes)

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_test_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model)
    elif cfg.MODEL.ADAPTATION == "gtta":
        model, param_names = setup_gtta(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "rmt":
        model, param_names = setup_rmt(base_model, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruption or domain names
    if "non_stationary" in cfg.SETTING:
        domain_sequence = ["mixed_domains"]
    elif cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
        logger.info(f"Using the following domain sequence: {domain_sequence}")
    else:
        domain_sequence = cfg.CORRUPTION.TYPE

    # evaluate on each severity and type of corruption in turn
    errs = []
    errs_5 = []
    for i_dom, domain_name in enumerate(domain_sequence):
        # continual adaptation for all corruption
        if i_dom == 0 or cfg.SETTING == "reset_each_shift":
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in cfg.CORRUPTION.SEVERITY:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               ckpt_path=cfg.CKPT_PATH,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=4)

            acc = get_accuracy(model, test_data_loader, dataset_name=cfg.CORRUPTION.DATASET)
            err = 1. - acc

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}]: {err:.2%}")
            errs.append(err)
            if severity == 5:
                errs_5.append(err)

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")


def setup_source(model):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model, None


def setup_test_norm(model):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    model.eval()
    for m in model.modules():
        # Re-activate batchnorm layer
        if (isinstance(m, nn.BatchNorm1d) and cfg.TEST.BATCH_SIZE > 1) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.train()

    # Wrap test normalization into Norm class to enable sliding window approach
    norm_model = Norm(model, optimizer=None, steps=1,
                      episodic=cfg.MODEL.EPISODIC,
                      window_length=cfg.TEST.WINDOW_LENGTH)
    return norm_model, None


def setup_alpha_norm(model):
    """Set up BN--0.1 (test-time normalization adaptation with source prior).
    Normalize features by combining the source moving statistics and the test batch statistics.
    """
    model.eval()
    norm_model = AlphaBatchNorm.adapt_model(model, alpha=cfg.BN.ALPHA).cuda()  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model, None


def setup_tent(model):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = Tent(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      window_length=cfg.TEST.WINDOW_LENGTH)
    return tent_model, param_names


def setup_cotta(model):
    model = CoTTA.configure_model(model)
    params, param_names = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = CoTTA(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        window_length=cfg.TEST.WINDOW_LENGTH,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP)
    return cotta_model, param_names


def setup_adacontrast(model):
    model = AdaContrast.configure_model(model)
    params, param_names = AdaContrast.collect_params(model)
    if cfg.CORRUPTION.DATASET == "domainnet126":
        optimizer = setup_adacontrast_optimizer(model)
    else:
        optimizer = setup_optimizer(params)

    adacontrast_model = AdaContrast(model, optimizer,
                                    steps=cfg.OPTIM.STEPS,
                                    episodic=cfg.MODEL.EPISODIC,
                                    dataset_name=cfg.CORRUPTION.DATASET,
                                    arch_name=cfg.MODEL.ARCH,
                                    queue_size=cfg.ADACONTRAST.QUEUE_SIZE,
                                    momentum=cfg.M_TEACHER.MOMENTUM,
                                    temperature=cfg.CONTRAST.TEMPERATURE,
                                    contrast_type=cfg.ADACONTRAST.CONTRAST_TYPE,
                                    ce_type=cfg.ADACONTRAST.CE_TYPE,
                                    alpha=cfg.ADACONTRAST.ALPHA,
                                    beta=cfg.ADACONTRAST.BETA,
                                    eta=cfg.ADACONTRAST.ETA,
                                    dist_type=cfg.ADACONTRAST.DIST_TYPE,
                                    ce_sup_type=cfg.ADACONTRAST.CE_SUP_TYPE,
                                    refine_method=cfg.ADACONTRAST.REFINE_METHOD,
                                    num_neighbors=cfg.ADACONTRAST.NUM_NEIGHBORS)
    return adacontrast_model, param_names


def setup_gtta(model, num_classes):
    model = GTTA.configure_model(model)
    params, param_names = GTTA.collect_params(model)
    optimizer = setup_optimizer(params)
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    _, src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                      root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                      batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, percentage=cfg.SOURCE.PERCENTAGE)
    gtta_model = GTTA(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      window_length=cfg.TEST.WINDOW_LENGTH,
                      dataset_name=cfg.CORRUPTION.DATASET,
                      num_classes=num_classes,
                      src_loader=src_loader,
                      ckpt_dir=cfg.CKPT_DIR,
                      ckpt_path=cfg.CKPT_PATH,
                      steps_adain=cfg.GTTA.STEPS_ADAIN,
                      pretrain_steps_adain=cfg.GTTA.PRETRAIN_STEPS_ADAIN,
                      style_transfer=cfg.GTTA.USE_STYLE_TRANSFER,
                      lam_mixup=cfg.GTTA.LAMBDA_MIXUP)
    return gtta_model, param_names


def setup_rmt(model, num_classes):
    model = RMT.configure_model(model)
    params, param_names = RMT.collect_params(model)
    optimizer = setup_optimizer(params)
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    _, src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                      root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                      batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, percentage=cfg.SOURCE.PERCENTAGE)
    rmt_model = RMT(model, optimizer,
                    steps=cfg.OPTIM.STEPS,
                    episodic=cfg.MODEL.EPISODIC,
                    window_length=cfg.TEST.WINDOW_LENGTH,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    arch_name=cfg.MODEL.ARCH,
                    num_classes=num_classes,
                    src_loader=src_loader,
                    ckpt_dir=cfg.CKPT_DIR,
                    ckpt_path=cfg.CKPT_PATH,
                    contrast_mode=cfg.CONTRAST.MODE,
                    temperature=cfg.CONTRAST.TEMPERATURE,
                    projection_dim=cfg.CONTRAST.PROJECTION_DIM,
                    lambda_ce_src=cfg.RMT.LAMBDA_CE_SRC,
                    lambda_ce_trg=cfg.RMT.LAMBDA_CE_TRG,
                    lambda_cont=cfg.RMT.LAMBDA_CONT,
                    m_teacher_momentum=cfg.M_TEACHER.MOMENTUM,
                    num_samples_warm_up=cfg.RMT.NUM_SAMPLES_WARM_UP)
    return rmt_model, param_names


def setup_optimizer(params):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_adacontrast_optimizer(model):
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": cfg.OPTIM.LR,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
                {
                    "params": extra_params,
                    "lr": cfg.OPTIM.LR * 10,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIM.METHOD} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


if __name__ == '__main__':
    evaluate('"Evaluation.')
