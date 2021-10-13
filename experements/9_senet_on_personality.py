import os
import torch.optim as optim
import torch.nn as nn
from dpcv.config.senet_cfg import cfg
from dpcv.modeling.module.se_resnet import se_resnet50
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.interpret_dan_data import make_data_loader
from dpcv.engine.bi_modal_trainer import InterpretDanTrain
from dpcv.tools.exp import run


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.OUTPUT_DIR)

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")

    model = se_resnet50(5)
    loss_f = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = InterpretDanTrain(cfg, collector, logger)

    run(cfg, train_loader, valid_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
