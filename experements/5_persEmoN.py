import os
import torch.optim as optim
from datetime import datetime
from dpcv.config.per_emo_cfg import cfg
from dpcv.modeling.networks.sphereface_net import get_pers_emo_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.pers_emo_data import make_data_loader
from dpcv.engine.bi_modal_trainer import PersEmoTrainer
from dpcv.modeling.loss.pers_emo_loss import per_emo_loss
from dpcv.tools.exp import run


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.OUTPUT_DIR)

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")

    model = get_pers_emo_model()
    loss_f = per_emo_loss

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = PersEmoTrainer(cfg, collector, logger)

    run(cfg, train_loader, valid_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)

    # if cfg.RESUME:
    #     model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
    #     cfg.START_EPOCH = epoch
    #     logger.info(f"resume training from {cfg.RESUME}")
    #
    # for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
    #     trainer.train(train_loader, model, loss_f, optimizer, epoch)
    #     trainer.valid(valid_loader, model, loss_f, epoch)
    #     scheduler.step()
    #     if collector.model_save:
    #         save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
    #         collector.update_best_epoch(epoch)
    #
    # collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    # logger.info(
    #     "{} done, best acc: {} in :{}".format(
    #         datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
    #         collector.best_valid_acc,
    #         collector.best_epoch,
    #     ))


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
