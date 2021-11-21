import torch.optim as optim
import torch.nn as nn
from dpcv.data.datasets.build import build_dataloader
from dpcv.config.senet_cfg import cfg
from dpcv.modeling.module.se_resnet import se_resnet50
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.video_frame_data import make_data_loader
from dpcv.engine.bi_modal_trainer import ImageModalTrainer
from dpcv.tools.exp import run


class ExpRunner:

    def __init__(self, cfg):
        """ construct certain experiment by the following template

        step 1: prepare dataloader
        step 2: prepare model and loss function
        step 3: select optimizer for gradient descent algorithm
        step 4: prepare trainer for typical training in pytorch manner
        """
        self.data_loader = self.get_dataloader(cfg)

        self.model = self.get_model(cfg)
        self.loss_f = self.get_loss_func(cfg)

        self.optimizer = self.get_optimizer(cfg)
        self.scheduler = self.get_scheduler(cfg)

        self.collector = TrainSummary()
        self.trainer = self.get_trainer(cfg)

    @staticmethod
    def get_dataloader(cfg):
        dataloader = build_dataloader(cfg)
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
            "test": dataloader(cfg, mode="test"),
        }
        return data_loader_dicts

# def main(args, cfg):
#     setup_seed(12345)
#     cfg = setup_config(args, cfg)
#     logger, log_dir = make_logger(cfg.OUTPUT_DIR)
#
#     data_loader = {
#         "train": make_data_loader(cfg, mode="train"),
#         "valid": make_data_loader(cfg, mode="valid"),
#         "test": make_data_loader(cfg, mode="test"),
#     }
#
#     model = se_resnet50(5)
#     loss_f = nn.MSELoss()
#
#     optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)
#
#     collector = TrainSummary()
#     trainer = ImageModalTrainer(cfg, collector, logger)
#
#     run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)


if __name__ == "__main__":
    args = parse_args()
    # main(args, cfg)
