import os
import json
import numpy as np
import torch
from datetime import datetime
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.engine.build import build_trainer
from dpcv.evaluation.summary import TrainSummary
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.evaluation.metrics import compute_pcc, compute_ccc
from dpcv.tools.logger import make_logger


class ExpRunner:

    def __init__(self, cfg, feature_extract=None):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """

        """
        construct certain experiment by the following template
        step 1: prepare dataloader
        step 2: prepare model and loss function
        step 3: select optimizer for gradient descent algorithm
        step 4: prepare trainer for typical training in pytorch manner
        """
        self.cfg = cfg
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        self.log_cfg_info()
        if not feature_extract:
            self.data_loader = self.build_dataloader()

        self.model = self.build_model()
        self.loss_f = self.build_loss_function()

        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()

        self.collector = TrainSummary()
        self.trainer = self.build_trainer()

    def build_dataloader(self):
        return build_dataloader(self.cfg)

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        return build_trainer(self.cfg, self.collector, self.logger)

    def before_train(self, cfg):
        # cfg = self.cfg.TRAIN
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            cfg.START_EPOCH = epoch
            self.logger.info(f"resume training from {cfg.RESUME}")
        if self.cfg.SOLVER.RESET_LR:
            self.logger.info("change learning rate form [{}] to [{}]".format(
                self.optimizer.param_groups[0]["lr"],
                self.cfg.SOLVER.LR_INIT,
            ))
            self.optimizer.param_groups[0]["lr"] = self.cfg.SOLVER.LR_INIT

    def train_epochs(self, cfg):
        # cfg = self.cfg.TRAIN
        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)
            if epoch % cfg.VALID_INTERVAL == 0:
                self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)
            self.scheduler.step()

            if self.collector.model_save and epoch % cfg.VALID_INTERVAL == 0:
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
                self.collector.update_best_epoch(epoch)
            if epoch == (cfg.MAX_EPOCH - 1):
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)

    def after_train(self, cfg):
        # cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        self.logger.info(
            "{} done, best acc: {} in :{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch,
            )
        )

    def train(self):
        cfg = self.cfg.TRAIN
        self.before_train(cfg)
        self.train_epochs(cfg)
        self.after_train(cfg)

    def test(self, weight=None):
        self.logger.info("Test only mode")
        cfg = self.cfg.TEST
        cfg.WEIGHT = weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            try:
                weights = [file for file in os.listdir(self.log_dir) if file.endswith(".pkl") and ("last" not in file)]
                weights = sorted(weights, key=lambda x: int(x[11:-4]))
                weight_file = os.path.join(self.log_dir, weights[-1])
            except IndexError:
                weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl")
            self.logger.info(f"test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        if not self.cfg.TEST.FULL_TEST:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label, mse = self.trainer.test(
                self.data_loader["test"], self.model
            )
            self.logger.info("mse: {} mean: {}".format(mse[0], mse[1]))
            self.latex_info(mse[0], mse[1])
        else:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(
                self.data_loader["full_test"], self.model
            )
        self.logger.info("acc: {} mean: {}".format(ocean_acc, ocean_acc_avg))
        self.latex_info(ocean_acc, ocean_acc_avg)  # a helper for latex table

        if cfg.COMPUTE_PCC:
            pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label)
            self.logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")
            self.latex_info(pcc_dict, pcc_mean)

        if cfg.COMPUTE_CCC:
            ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label)
            self.logger.info(f"ccc: {ccc_dict} mean: {ccc_mean}")
            self.latex_info(ccc_dict, ccc_mean)
        if cfg.SAVE_DATASET_OUTPUT:
            os.makedirs(cfg.SAVE_DATASET_OUTPUT, exist_ok=True)
            torch.save(dataset_output, os.path.join(cfg.SAVE_DATASET_OUTPUT, "pred.pkl"))
            torch.save(dataset_label, os.path.join(cfg.SAVE_DATASET_OUTPUT, "label.pkl"))
        return

    def run(self):
        self.train()
        self.test()

    @staticmethod
    def latex_info(metric, mean):
        latex_tab = ""
        if isinstance(metric, dict):
            for k, v in metric.items():
                latex_tab += str(v) + " & "
        else:
            for v in metric:
                latex_tab += str(np.round(v, 4)) + " & "
        latex_tab += str(mean)
        print(latex_tab)

    def log_cfg_info(self):
        """
        record training info for convenience of results analysis
        """
        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):

        return self.trainer.data_extract(self.model, dataloader, output_dir)


