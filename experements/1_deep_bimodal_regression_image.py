import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import argparse
import torch.optim as optim
from datetime import datetime
from dpcv.engine.bi_modal_trainer import BiModalTrainer
from dpcv.tools.common import setup_seed
from dpcv.tools.draw import plot_line
from dpcv.tools.logger import make_logger
from dpcv.modeling.networks.dan import get_dan_model
from dpcv.config.deep_bimodal_regression_cfg import cfg
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.data.datasets.chalearn_data import make_data_loader
from dpcv.evaluation.summary import TrainSummary

def main():
    setup_seed(12345)

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

    cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
    cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
    cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
    cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create logger
    logger, log_dir = make_logger("../results")

    # step1： get dataset
    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="val")

    # step2: set model
    model = get_dan_model(pretrained=True)

    # step3: loss functions and optimizer
    loss_f = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = BiModalTrainer(cfg, collector, logger)

    start_epoch = cfg.START_EPOCH
    if args.resume:
        model, optimizer, epoch = resume_training(args.resume, model, optimizer)
        start_epoch = epoch
        logger.info(f"resume training from {args.resume}")

    for epoch in range(start_epoch, cfg.MAX_EPOCH):
        # train for one epoch
        trainer.train(train_loader, model, loss_f, optimizer, epoch)
        # eval after training an epoch
        trainer.valid(valid_loader, model, loss_f, epoch)
        # update training lr every epoch
        scheduler.step()

        # save model
        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - start_epoch, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch)
    )


if __name__ == "__main__":
    main()
