from dpcv.tools.common import parse_args
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list
from dpcv.experiment.exp_runner import ExpRunner


def setup():
    args = parse_args()
    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    # cfg_from_list(args.opts)


if __name__ == "__main__":
    import os
    os.chdir("..")
    setup()
    runner = ExpRunner(cfg)
    runner.run()
    # runner.test("results/senet/10-14_02-34/checkpoint_295.pkl")
