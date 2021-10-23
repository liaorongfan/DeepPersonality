import pprint
from datetime import datetime
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate
import torch
from dpcv.checkpoint.save import save_model, resume_training, load_model
from torch.nn import Module


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "outputs"
        self.print_interval = 100
        self.eval_interval = 10

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_train_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_valid_data_loader(
            self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ):
        pass

    # @abstractmethod
    # def run(self):
    #     pass

    # @abstractmethod
    # def get_evaluator(self):
    #     pass

    @abstractmethod
    def test(self, model, evaluator, weights):
        pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    # def merge(self, cfg_list):
    #     assert len(cfg_list) % 2 == 0
    #     for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    #         # only update value with same key
    #         if hasattr(self, k):
    #             src_value = getattr(self, k)
    #             src_type = type(src_value)
    #             if src_value is not None and src_type != type(v):
    #                 try:
    #                     v = src_type(v)
    #                 except Exception:
    #                     v = ast.literal_eval(v)
    #             setattr(self, k, v)


def run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir):

    if cfg.TEST_ONLY:
        from scipy.stats import pearsonr
        model = load_model(model, cfg.WEIGHT)
        ocean_acc_avg, ocean_acc, dataset_output, dataset_label = trainer.test(data_loader["test"], model)
        # ocean_acc_avg, ocean_acc = trainer.test(test_loader, model)
        pcc = pearsonr(dataset_output, dataset_label)
        logger.info(f"average acc of OCEAN:{ocean_acc},\taverage acc [{ocean_acc_avg}]\npcc and p_value:{pcc}")
        return

    if cfg.RESUME:
        model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
        cfg.START_EPOCH = epoch
        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
        trainer.train(data_loader["train"], model, loss_f, optimizer, epoch)
        trainer.valid(data_loader["valid"], model, loss_f, epoch)
        scheduler.step()

        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch
        )
    )

