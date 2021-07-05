import os
import torch
import random
import numpy as np
import argparse


def setup_config(args, cfg):
    cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
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
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=None, help='learning rate')
    parser.add_argument('--bs', default=None, help='training batch size')
    parser.add_argument("--resume", default=None,
                        help="saved model path to last training epoch")
    parser.add_argument('--max_epoch', default=None)
    parser.add_argument('--data_root_dir',
                        default=r"/home/rongfan/11-personality_traits/DeepPersonality/datasets",
                        help="path to your dataset")
    args = parser.parse_args()
    return args

