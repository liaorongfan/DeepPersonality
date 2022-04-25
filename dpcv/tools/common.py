import os
import torch
import random
import numpy as np
import argparse


def setup_config(args, cfg):
    # cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
    cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
    cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
    cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH
    cfg.RESUME = args.resume if args.resume else cfg.RESUME
    return cfg


def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='deep learning on personality')
    parser.add_argument(
        '-c',
        '--cfg_file',
        help="experiment config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        '--weight',
        dest='weight',
        help='initialize with pretrained model weights',
        type=str,
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="only test model on specified weights",
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='learning rate',
    )
    parser.add_argument(
        '--bs',
        default=None,
        help='training batch size',
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="saved model path to last training epoch",
    )
    parser.add_argument(
        "-m",
        '--max_epoch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def get_device(gpu=0):
    device = torch.device(
        f'cuda:{gpu}'
        if torch.cuda.is_available() and gpu is not None
        else 'cpu')
    return device


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output
