#! /usr/bin/env python

from dpcv.tools.common import parse_args
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list
from torch.utils.tensorboard import SummaryWriter
from dpcv.experiment.exp_runner import ExpRunner


def setup():
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.max_epoch:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch
    if args.lr:
        cfg.SOLVER.RESET_LR = True
        cfg.SOLVER.LR_INIT = args.lr
    if args.test_only:
        cfg.TEST.TEST_ONLY = True

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args


def main():
    args = setup()
    runner = ExpRunner(cfg)
    if args.test_only:
        return runner.test()
    runner.run()


if __name__ == "__main__":
    # for debug setting
    import os
    # os.chdir("..")
    main()
