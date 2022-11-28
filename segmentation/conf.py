# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import math
import torch
import glob
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output"

# Name of experiment
_C.EXP_NAME = ""

# Save predicted images
_C.SAVE_PREDICTIONS = False

# Path to file containing the source images
_C.LIST_NAME_SRC = "clear_train.txt"

# Path to file containing the test images
_C.LIST_NAME_TEST = "day_night_1200.txt"

# Path to a pre-trained segmentation model
_C.CKPT_PATH_SEG = "./ckpt/clear/ckpt_seg.pth"

# Path to a pre-trained style transfer model (based on AdaIN)
_C.CKPT_PATH_ADAIN_DEC = "./ckpt/clear/ckpt_adain.pth"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 12345

# Deterministic experiments.
_C.DETERMINISM = False

# Optional description of a config
_C.DESC = ""

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Choice of (source, norm, tent)
_C.MODEL.ADAPTATION = "source"

# Name of the architecture
_C.MODEL.NAME = "deeplabv2"

# Number of casses
_C.MODEL.NUM_CLASSES = 14

# Initialize model weights with imagenet pre-trained weights. If a checkpoint
# path is defined in 'CKPT_PATH_SEG', this has no further effect.
_C.MODEL.IMAGENET_INIT = True

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation
_C.TEST.BATCH_SIZE = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 6

# Smaller side of test image
_C.TEST.IMG_SIZE = 1024

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Name of the optimizer. Choices: SGD
_C.OPTIM.METHOD = 'SGD'

# Learning rate
_C.OPTIM.LR = 2.5e-4

# Factor used to scale the learning rate of the segmentation head
_C.OPTIM.SCALE_LR_SEGHEAD = 10

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Regularization parameter for weight decay.
_C.OPTIM.WD = 0.0005

# Momentum component of SGD optimizer
_C.OPTIM.MOMENTUM = 0.9

# Nesterov momentum
_C.OPTIM.NESTEROV = False

# Number of workers for data loading
_C.OPTIM.WORKERS = 4

# Label to ignore during optimization
_C.OPTIM.IGNORE_LABEL = 255


# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.9

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.USE_STYLE_TRANSFER = True
_C.GTTA.LAMBDA_CE_TRG = 0.1


# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Percentage of source samples used during training.
# [0, 1] possibility to reduce the number of source samples
_C.SOURCE.PERCENTAGE = 1.0

# Whether to randomly crop images during training
_C.SOURCE.RANDOM_CROP = False

# Whether to randomly scale images before cropping them
_C.SOURCE.RANDOM_SCALE_CROP = True

# Probability to apply left right flipping. (Not applied for value 0.)
_C.SOURCE.PROB_FLIP = 0.5

# Probability to apply gaussian blur (Not applied for value 0.)
_C.SOURCE.PROB_BLUR = 0.5

# Probability to apply color jittering (Not applied for value 0.)
_C.SOURCE.PROB_JITTER = 0.8

# Value used for changing the: brightness, contrast, saturation, hue
_C.SOURCE.JITTER_VAL = 0.25

# Miniumum scaling of base_size if scale_crop is used
_C.SOURCE.MIN_SCALE = 0.75

# Maximum scaling of base_size if scale_crop is used
_C.SOURCE.MAX_SCALE = 2.0

# Resize smaller side of the image to this size before doing cropping
_C.SOURCE.BASE_SIZE = 512

# Size to crop the image to
_C.SOURCE.CROP_SIZE = [1024, 512]

# Number of samples in a source batch
_C.SOURCE.BATCH_SIZE = 2

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def increment_path(dir_path, name):
    if os.path.exists(dir_path):
        num = len([item for item in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, item)) and name in item])
    else:
        num = 0
    return name + f'_{num}'


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.EXP_NAME = increment_path(cfg.SAVE_DIR, cfg.EXP_NAME + f"_{cfg.LIST_NAME_TEST.split('_')[0]}" if cfg.EXP_NAME else cfg.MODEL.ADAPTATION)
    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def map_asm_rain_paths(ckpt_dir, model_name):
    model2file_name = {'vgg_enc_path': os.path.join(ckpt_dir, "pretrained_rain", "vgg_normalised.pth"),
                       'vgg_dec_path':  os.path.join(ckpt_dir, "pretrained_rain", "decoder_iter_160000.pth"),
                       'style_enc_path': os.path.join(ckpt_dir, "pretrained_rain", "fc_encoder_iter_160000.pth"),
                       'style_dec_path': os.path.join(ckpt_dir, "pretrained_rain", "fc_decoder_iter_160000.pth")}
    return model2file_name[model_name]
