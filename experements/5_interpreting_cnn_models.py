import torch.nn as nn
import os
import torch.optim as optim
from datetime import datetime
from dpcv.config.interpret_dan_cfg import cfg
from dpcv.engine.bi_modal_trainer import InterpretDanTrain
from dpcv.modeling.networks.interpret_dan import get_interpret_dan_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.interpret_dan_data import make_data_loader


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    res_dir = os.path.join("..", "results")
    logger, log_dir = make_logger(res_dir)
    logger.info("file_name: \n{}\n".format(__file__))

    train_loader = make_data_loader(cfg, mode="valid")
    valid_loader = make_data_loader(cfg, mode="test")

    model = get_interpret_dan_model(cfg, pretrained=True)
    loss_f = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = InterpretDanTrain(cfg, collector, logger)

    # start_epoch = cfg.START_EPOCH
    if cfg.RESUME:
        model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
        cfg.START_EPOCH = epoch
        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
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

    collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
            collector.best_valid_acc,
            collector.best_epoch,
        ))


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
