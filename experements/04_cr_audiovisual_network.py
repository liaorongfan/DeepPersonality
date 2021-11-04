import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from dpcv.config.crnet_cfg import cfg as cr_cfg
from dpcv.engine.crnet_trainer import CRNetTrainer
from dpcv.tools.logger import make_logger
from dpcv.modeling.networks.cr_net import get_crnet_model
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.data.datasets.cr_data import make_data_loader
from dpcv.tools.common import parse_args, setup_config, setup_seed
from dpcv.evaluation.summary import TrainSummary
from dpcv.modeling.loss.cr_loss import one_hot_CELoss, BellLoss
from dpcv.evaluation.metrics import compute_ccc, compute_pcc


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.OUTPUT_DIR)

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")
    test_loader = make_data_loader(cfg, mode="test")

    model = get_crnet_model(only_train_guider=True)
    loss_f = {"ce_loss": one_hot_CELoss, "bell_loss": BellLoss(), "mse_loss": nn.MSELoss(), "l1_loss": nn.L1Loss()}

    optimizer_fir = optim.SGD(model.parameters(), lr=cfg.LR_INIT, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer_sec = optim.Adam(
        model.parameters(), betas=(cfg.BETA_1, cfg.BETA_2), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY
    )
    optimizer = [optimizer_fir, optimizer_sec]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sec, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = CRNetTrainer(cfg, collector, logger)

    if cfg.TEST_ONLY:
        model.train_regressor()
        model = load_model(model, cfg.WEIGHT)
        ocean_acc_avg, ocean_acc, dataset_output, dataset_label = trainer.test(test_loader, model)
        pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label)
        ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label)
        logger.info(f"acc: {ocean_acc} mean: {ocean_acc_avg}")
        logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")
        logger.info(f"acc: {ccc_dict} mean: {ccc_mean}")
        return

    if cfg.RESUME:
        model, optimizer[1], epoch = resume_training(cfg.RESUME, model, optimizer[1])
        cfg.START_EPOCH = epoch

        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(cfg.START_EPOCH, cfg.TRAIN_CLS_EPOCH):
        if cfg.START_EPOCH > cfg.TRAIN_CLS_EPOCH:
            break
        model.train_classifier()
        trainer.train(train_loader, model, loss_f, optimizer, epoch)

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
        model.train_regressor()
        # train for one epoch
        trainer.train(train_loader, model, loss_f, optimizer, epoch)
        # eval after training an epoch
        trainer.valid(valid_loader, model, loss_f, epoch)
        # update training lr every epoch
        scheduler.step()
        # save model
        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer[1], log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch)
    )


if __name__ == "__main__":
    args = parse_args()
    main(args, cr_cfg)
