import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import argparse
import torch.optim as optim
from datetime import datetime
from dpcv.engine.train import BiModalTrainer
from dpcv.utiles.common import setup_seed
from dpcv.utiles.draw import plot_line
from dpcv.utiles.logger import make_logger
from dpcv.modeling.networks.bi_modal_lstm import get_bi_modal_lstm_model
from dpcv.config.deep_bimodal_regression_cfg import cfg
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.data.datasets.temporal_data import make_data_loader

# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, '..'))


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
    res_dir = os.path.join("..", "results")
    logger, log_dir = make_logger(res_dir)

    # step1： get dataset
    train_loader = make_data_loader(cfg, mode="train")
    # valid_loader = make_data_loader(cfg, mode="val")

    # step2: set model
    model = get_bi_modal_lstm_model()

    # step3: loss functions and optimizer
    loss_f = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)
    start_epoch = 1
    if args.resume:
        model, optimizer, epoch = resume_training(args.resume, model, optimizer)
        start_epoch = epoch
        logger.info(f"resume training from {args.resume}")
    # step4: training iteratively
    # record info including model structure, loss function, optimizer and configs
    logger.info(
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
            cfg, loss_f, scheduler, optimizer, model
        )
    )

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    batch_loss = []
    batch_acc = []
    best_acc, best_epoch = 0, 0
    for epoch in range(start_epoch, cfg.MAX_EPOCH + 1):
        # train for one epoch
        loss_train, acc_train, loss_list, acc_avg_list = BiModalTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger
        )
        # loss_train = 0
        # acc_train = 0

        # eval for after training for a epoch
        # loss_valid, ocean_acc_valid, acc_avg_valid = ModelTrainer.valid(
        #     valid_loader, model, loss_f, device
        # )
        loss_valid = 0; acc_avg_valid = 0; ocean_acc_valid = 0
        # display info for that training epoch
        logger.info(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{} \n"
            "Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
                epoch + 1, cfg.MAX_EPOCH, acc_train, float(acc_avg_valid), ocean_acc_valid,
                loss_train, loss_valid,
                optimizer.param_groups[0]["lr"]
            )
        )
        # update training lr every epoch
        scheduler.step()

        # save model
        if acc_avg_valid > best_acc:
            save_model(epoch, best_acc, model, optimizer, log_dir, cfg)
            best_epoch = epoch
            best_acc = acc_avg_valid

        # plot loss and acc every epoch
        # loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        # acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_avg_valid)
        # batch_loss.extend(loss_list), batch_acc.extend(acc_avg_list)
        #
        # plt_x = np.arange(1, epoch + 2)
        # plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        # plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        #
        # plt_x_batch = np.arange(1, len(batch_loss) + 1)
        # plot_line(plt_x_batch, batch_loss, plt_x_batch, batch_acc, mode="batch info", out_dir=log_dir)

    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch
        )
    )


if __name__ == "__main__":
    main()
