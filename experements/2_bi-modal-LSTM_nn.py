import torch.nn as nn
import os
import torch.optim as optim
from dpcv.config.bi_modal_lstm_cfg import cfg
from dpcv.engine.bi_modal_trainer import BimodalLSTMTrain, ImgModalLSTMTrain, AudModalLSTMTrain
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.modeling.networks.bi_modal_lstm import (
    get_bi_modal_lstm_model,
    get_img_modal_lstm_model,
    get_aud_modal_lstm_model
)
from dpcv.data.datasets.temporal_data import make_data_loader
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.tools.exp import run


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.OUTPUT_DIR)

    data_loader = {
        "train": make_data_loader(cfg, mode="train"),
        "valid": make_data_loader(cfg, mode="valid"),
        "test": make_data_loader(cfg, mode="test"),
    }

    model = get_bi_modal_lstm_model()
    # model = get_img_modal_lstm_model()  # to test single performance
    # model = get_aud_modal_lstm_model()  # to test single performance
    loss_f = nn.MSELoss()  # according to the paper
    # loss_f = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = BimodalLSTMTrain(cfg, collector, logger)
    # trainer = ImgModalLSTMTrain(cfg, collector, logger)
    # trainer = AudModalLSTMTrain(cfg, collector, logger)

    run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)
    # if cfg.RESUME:
    #     model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
    #     cfg.START_EPOCH = epoch
    #     logger.info(f"resume training from {cfg.RESUME}")
    #
    # for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
    #     trainer.train(train_loader, model, loss_f, optimizer, epoch)
    #     trainer.valid(valid_loader, model, loss_f, epoch)
    #     scheduler.step()
    #
    #     if collector.model_save:
    #         save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
    #         collector.update_best_epoch(epoch)
    #
    # collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    # logger.info(
    #     "{} done, best acc: {} in :{}".format(
    #         datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch
    #     )
    # )


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
