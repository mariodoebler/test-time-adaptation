
import os
import logging
import math
import numpy as np
import torch.optim as optim
import torch.nn as nn

from models.model import get_model
from utils import get_accuracy, eval_domain_dict
from conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
from datasets.data_loading import get_source_loader, get_test_loader

from methods.bn import AlphaBatchNorm, EMABatchNorm
from methods.tent import Tent
from methods.ttaug import TTAug
from methods.memo import MEMO
from methods.cotta import CoTTA
from methods.gtta import GTTA
from methods.adacontrast import AdaContrast
from methods.rmt import RMT
from methods.eata import EATA
from methods.norm import Norm
from methods.lame import LAME
from methods.sar import SAR, SAM
from methods.rotta import RoTTA


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    assert cfg.SETTING in ["reset_each_shift",           # reset the model state after the adaptation to a domain
                           "continual",                  # train on sequence of domain shifts without knowing when shift occurs
                           "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                           "mixed_domains",              # consecutive test samples are likely to originate from different domains
                           "correlated",                 # sorted by class label
                           "mixed_domains_correlated",   # mixed domains + sorted by class label
                           "gradual_correlated",         # gradual domain shifts + sorted by class label
                           "reset_each_shift_correlated"
                           ]

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes)

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_test_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "norm_ema":  # BN--EMA
        model, param_names = setup_ema_norm(base_model)
    elif cfg.MODEL.ADAPTATION == "ttaug":
        model, param_names = setup_ttaug(base_model)
    elif cfg.MODEL.ADAPTATION == "memo":
        model, param_names = setup_memo(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model)
    elif cfg.MODEL.ADAPTATION == "lame":
        model, param_names = setup_lame(base_model)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model)
    elif cfg.MODEL.ADAPTATION == "eta":
        model, param_names = setup_eta(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "eata":
        model, param_names = setup_eata(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "sar":
        model, param_names = setup_sar(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "rotta":
        model, param_names = setup_rotta(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "gtta":
        model, param_names = setup_gtta(base_model, num_classes)
    elif cfg.MODEL.ADAPTATION == "rmt":
        model, param_names = setup_rmt(base_model, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               domain_names_all=dom_names_all,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            acc, domain_dict = get_accuracy(
                model, data_loader=test_data_loader, dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name, setting=cfg.SETTING, domain_dict=domain_dict)

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=cfg.CORRUPTION.TYPE)


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
        if (isinstance(m, nn.BatchNorm1d) and cfg.TEST.BATCH_SIZE > 1) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
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


def setup_ema_norm(model):
    """Set up BN--EMA (test-time normalization adaptation using an exponential moving average).
    """
    norm_model = EMABatchNorm.adapt_model(model).cuda()
    return norm_model, None


def setup_ttaug(model):
    model = AlphaBatchNorm.adapt_model(model, alpha=cfg.BN.ALPHA)
    ttaug_model = TTAug(model, None,
                        steps=1,
                        episodic=cfg.MODEL.EPISODIC,
                        n_augmentations=cfg.TEST.N_AUGMENTATIONS,
                        dataset_name=cfg.CORRUPTION.DATASET)
    return ttaug_model, None


def setup_memo(model):
    model = AlphaBatchNorm.adapt_model(model, alpha=cfg.BN.ALPHA)
    params, param_names = MEMO.collect_params(model)
    optimizer = setup_optimizer(params)
    memo_model = MEMO(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      n_augmentations=cfg.TEST.N_AUGMENTATIONS,
                      dataset_name=cfg.CORRUPTION.DATASET)
    return memo_model, param_names


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


def setup_lame(model):
    model = LAME.configure_model(model)
    lame_model = LAME(model, optimizer=None,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      window_length=cfg.TEST.WINDOW_LENGTH,
                      dataset_name=cfg.CORRUPTION.DATASET,
                      arch_name=cfg.MODEL.ARCH,
                      affinity=cfg.LAME.AFFINITY,
                      knn=cfg.LAME.KNN,
                      sigma=cfg.LAME.SIGMA,
                      force_symmetry=cfg.LAME.FORCE_SYMMETRY)
    return lame_model, None


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


def setup_eta(model, num_classes):
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    optimizer = setup_optimizer(params)
    eta_model = EATA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     window_length=cfg.TEST.WINDOW_LENGTH,
                     e_margin=math.log(num_classes)*0.40,
                     d_margin=cfg.EATA.D_MARGIN)
    return eta_model, param_names


def setup_eata(model, num_classes):
    # compute fisher informatrix
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    _, fisher_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                         root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                         batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, num_samples=cfg.EATA.NUM_SAMPLES)

    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, batch in enumerate(fisher_loader, start=1):
        images = batch[0].cuda(non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    logger.info("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = setup_optimizer(params)
    eta_model = EATA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     window_length=cfg.TEST.WINDOW_LENGTH,
                     fishers=fishers,
                     fisher_alpha=cfg.EATA.FISHER_ALPHA,
                     e_margin=math.log(num_classes)*0.40,
                     d_margin=cfg.EATA.D_MARGIN)

    return eta_model, param_names


def setup_sar(model, num_classes):
    model = SAR.configure_model(model)
    params, param_names = SAR.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    sar_model = SAR(model, optimizer,
                    steps=cfg.OPTIM.STEPS,
                    episodic=cfg.MODEL.EPISODIC,
                    window_length=cfg.TEST.WINDOW_LENGTH,
                    margin_e0=math.log(num_classes)*0.40,
                    reset_constant_em=cfg.SAR.RESET_CONSTANT_EM)
    return sar_model, param_names


def setup_rotta(model, num_classes):
    model = RoTTA.configure_model(model, alpha=cfg.ROTTA.ALPHA)
    params, param_names = RoTTA.collect_params(model)
    optimizer = setup_optimizer(params)
    sar_model = RoTTA(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      window_length=cfg.TEST.WINDOW_LENGTH,
                      dataset_name=cfg.CORRUPTION.DATASET,
                      memory_size=cfg.ROTTA.MEMORY_SIZE,
                      num_classes=num_classes,
                      lambda_t=cfg.ROTTA.LAMBDA_T,
                      lambda_u=cfg.ROTTA.LAMBDA_U,
                      nu=cfg.ROTTA.NU,
                      update_freq=cfg.ROTTA.UPDATE_FREQUENCY)
    return sar_model, param_names


def setup_gtta(model, num_classes):
    model = GTTA.configure_model(model)
    params, param_names = GTTA.collect_params(model)
    optimizer = setup_optimizer(params)
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    _, src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                      root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                      batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, percentage=cfg.SOURCE.PERCENTAGE,
                                      workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
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
                                      batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH, percentage=cfg.SOURCE.PERCENTAGE,
                                      workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
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
