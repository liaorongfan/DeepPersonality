import torch.optim as optim
from dpcv.config.per_emo_cfg import cfg
from dpcv.modeling.networks.sphereface_net import get_pers_emo_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
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

    data_loader = {
        "train": make_data_loader(cfg, mode="train"),
        "valid": make_data_loader(cfg, mode="valid"),
        "test": make_data_loader(cfg, mode="test"),
    }

    model = get_pers_emo_model()
    loss_f = per_emo_loss

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = PersEmoTrainer(cfg, collector, logger)

    run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
