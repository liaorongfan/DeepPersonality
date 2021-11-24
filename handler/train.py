import argparse
# from dpcv.tools.common import parse_args
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list

from dpcv.experiment.exp_runner import ExpRunner


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("-c", "--config_file", default=None, help="config file for experiment")
    parser.add_argument('--lr', default=None, help='learning rate')
    parser.add_argument('--bs', default=None, help='training batch size')
    parser.add_argument("--resume", default=None, help="saved model path to last training epoch")
    parser.add_argument('--max_epoch', default=None)
    args = parser.parse_args()
    return args


def setup():
    args = parse_args()
    if args.config_file:
        cfg_from_file(args.config_file)
    # cfg_from_list(args.opts)


if __name__ == "__main__":
    import os
    os.chdir("..")
    setup()
    runner = ExpRunner(cfg)
    runner.run()
    # runner.test("results/senet/10-14_02-34/checkpoint_295.pkl")
