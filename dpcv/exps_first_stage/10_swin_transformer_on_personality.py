import torch.optim as optim
import torch.nn as nn
from dpcv.config.swin_transformer_cfg import cfg
from dpcv.modeling.networks.swin_transformer import get_swin_transformer_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.video_frame_data import make_data_loader
from dpcv.engine.bi_modal_trainer import ImageModalTrainer
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

    model = get_swin_transformer_model()
    loss_f = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = ImageModalTrainer(cfg, collector, logger)

    run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)
