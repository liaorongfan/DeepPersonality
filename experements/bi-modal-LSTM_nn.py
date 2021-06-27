import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from datetime import datetime
from dpcv.config.bi_modal_lstm_cfg import cfg
from dpcv.engine.bi_modal_lstm_train import BiModalTrainer
from dpcv.tools.common import setup_seed
from dpcv.tools.logger import make_logger
from dpcv.modeling.networks.bi_modal_lstm import get_bi_modal_lstm_model
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.data.datasets.temporal_data import make_data_loader
from dpcv.tools.common import parse_args
from dpcv.tools.common import TrainSummary


def setup_config():
    cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
    cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
    cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
    cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH
    return cfg


def main(cfg, args):
    setup_seed(12345)
    res_dir = os.path.join("..", "results")
    logger, log_dir = make_logger(res_dir)
    logger.info("file_name: \n{}\ncfg:\n{}\n".format(__file__, cfg))

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")

    model = get_bi_modal_lstm_model()
    loss_f = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = BiModalTrainer(collector)

    start_epoch = cfg.START_EPOCH
    if args.resume:
        model, optimizer, epoch = resume_training(args.resume, model, optimizer)
        start_epoch = epoch
        logger.info(f"resume training from {args.resume}")

    for epoch in range(start_epoch, cfg.MAX_EPOCH + 1):
        # train for one epoch
        trainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, cfg, logger
        )
        # eval after training an epoch
        trainer.valid(
            valid_loader, model, loss_f
        )
        # display info for that training epoch
        logger.info(
            "Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n Current lr:{} \n".format(
                epoch + 1, cfg.MAX_EPOCH,
                float(collector.mean_train_acc),
                float(collector.mean_valid_acc),
                collector.valid_ocean_acc,
                optimizer.param_groups[0]["lr"])
        )
        # update training lr every epoch
        scheduler.step()

        # save model
        if collector.model_save:
            save_model(epoch, collector.best_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_acc, collector.best_epoch)
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = setup_config()
    main(cfg, args)
