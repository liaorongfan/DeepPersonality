import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.config.default_config_opt import cfg
from dpcv.modeling.module.se_resnet import se_resnet50
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.video_frame_data import make_data_loader
from dpcv.engine.bi_modal_trainer import ImageModalTrainer
from dpcv.tools.exp import run
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.evaluation.metrics import compute_pcc, compute_ccc
from dpcv.tools.logger import make_logger


class ExpRunner:

    def __init__(self, cfg):
        """ construct certain experiment by the following template

        step 1: prepare dataloader
        step 2: prepare model and loss function
        step 3: select optimizer for gradient descent algorithm
        step 4: prepare trainer for typical training in pytorch manner
        """
        self.logger, self.log_dir = make_logger(cfg.OUTPUT_DIR)

        self.data_loader = self.build_dataloader(cfg)

        self.model = self.build_model(cfg)
        self.loss_f = self.build_loss_function(cfg)

        self.optimizer = self.build_solver(cfg, self.model)
        self.scheduler = self.build_scheduler(cfg, self.optimizer)

        self.collector = TrainSummary()
        self.trainer = self.build_trainer(cfg, self.collector, self.logger)


    @classmethod
    def build_dataloader(cls, cfg):
        dataloader = build_dataloader(cfg)
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
            "test": dataloader(cfg, mode="test"),
        }
        return data_loader_dicts

    @classmethod
    def build_model(cls, cfg):
        return build_model(cfg)

    @classmethod
    def build_loss_function(cls, cfg):
        return build_loss_func(cfg)

    @classmethod
    def build_solver(cls, cfg, model):
        return build_solver(cfg, model)

    @classmethod
    def build_scheduler(cls, cfg, optimizer):
        return build_scheduler(cfg, optimizer)

    @classmethod
    def build_trainer(cls, collector, logger):
        return

    def train(self, cfg):
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            cfg.START_EPOCH = epoch
            self.logger.info(f"resume training from {cfg.RESUME}")

        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)
            self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)
            self.scheduler.step()

            if self.collector.model_save:
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, log_dir, cfg)
                self.collector.update_best_epoch(epoch)

        self.collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
        self.logger.info(
            "{} done, best acc: {} in :{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch,
            )
        )

    def test(self, cfg):
        self.logger.info("Test only mode")
        self.model = load_model(self.model, cfg.WEIGHT)
        ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.test(
            self.data_loader["test"], self.model
        )
        self.logger.info("acc: {} mean: {}".format(ocean_acc, ocean_acc_avg))

        if cfg.COMPUTE_PCC:
            pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label)
            self.logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")

        if cfg.COMPUTE_CCC:
            ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label)
            self.logger.info(f"ccc: {ccc_dict} mean: {ccc_mean}")
        return


if __name__ == "__main__":
    # args = parse_args()
    import os
    import torch
    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")

    exp_runner = ExpRunner(cfg)
    xin = torch.randn((1, 3, 224, 224)).cuda()
    y = exp_runner.model(xin)
    print(y.shape)
    # main(args, cfg)
