
# def main(analysis_bad_case=False):
#     setup_seed(12345)
#
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('--lr', default=None, help='learning rate')
#     parser.add_argument('--bs', default=None, help='training batch size')
#     parser.add_argument('--max_epoch', default=None)
#     parser.add_argument('--data_root_dir',
#                         default=r"/home/rongfan/11-personality_traits/DeepPersonality/datasets/portrait",
#                         help="path to your dataset")
#     args = parser.parse_args()
#
#     cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
#     cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
#     cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
#     cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # create logger
#     res_dir = os.path.join("..", "results")
#     logger, log_dir = make_logger(res_dir)
#
#     # step1： get dataset
#     train_loader = make_data_loader(cfg, mode="train")
#     valid_loader = make_data_loader(cfg, mode="test")
#
#     # step2: set model
#     model = get_portrait_model()
#
#     # step3: loss functions and optimizer
#     loss_f = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)
#
#     # step4: training iteratively
#     # record info including model structure, loss function, optimizer and configs
#     logger.info(
#         "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
#             cfg, loss_f, scheduler, optimizer, model
#         )
#     )
#
#     loss_rec = {"train": [], "valid": []}
#     acc_rec = {"train": [], "valid": []}
#     best_acc, best_epoch = 0, 0
#     for epoch in range(cfg.MAX_EPOCH):
#         # train for one epoch
#         loss_train, acc_train, loss_list, acc_avg_list = ModelTrainer.train(
#             train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger
#         )
#         # loss_train = 0
#         # acc_train = 0
#         # eval for after training for a epoch
#         loss_valid, ocean_acc_valid, acc_avg_valid = ModelTrainer.valid(
#             valid_loader, model, loss_f, device
#         )
#         # display info for that training epoch
#         logger.info(
#             "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{} \n"
#             "Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
#                 epoch + 1, cfg.MAX_EPOCH, acc_train, float(acc_avg_valid), ocean_acc_valid,
#                 loss_train, loss_valid,
#                 optimizer.param_groups[0]["lr"]
#             )
#         )
#         # update training lr every epoch
#         scheduler.step()
#
#         # save model
#         if acc_avg_valid > best_acc:
#             save_model(epoch, best_acc, model, optimizer, log_dir, cfg)
#             best_epoch = epoch
#             best_acc = acc_avg_valid
#
#         # plot loss and acc every epoch
#         loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
#         acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_avg_valid)
#
#         plt_x = np.arange(1, epoch + 2)
#         plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
#         plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
#
#     logger.info(
#         "{} done, best acc: {} in :{}".format(
#             datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch
#         )
#     )
#

import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from dpcv.engine.bi_modal_trainer import BiModalTrainer
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.modeling.networks.audio_visual_residual import get_audiovisual_resnet_model
from dpcv.config.audiovisual_resnet_cfg import cfg
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.data.datasets.audio_visual_data import make_data_loader
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from scipy.stats import pearsonr


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.RESULT_DIR)
    logger.info("file_name: \n{}\n".format(__file__))

    collector = TrainSummary()
    trainer = BiModalTrainer(cfg, collector, logger)

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")
    # test_loader = make_data_loader(cfg, mode="test")

    model = get_audiovisual_resnet_model()
    if cfg.TEST_ONLY:
        model = load_model(model, cfg.WEIGHT)
        # ocean_acc_avg, ocean_acc, dataset_output, dataset_label = trainer.test(test_loader, model)
        ocean_acc_avg, ocean_acc = trainer.test(test_loader, model)
        # pcc = pearsonr(dataset_output, dataset_label)
        pcc = 0
        logger.info(f"average acc of OCEAN:{ocean_acc},\taverage acc [{ocean_acc_avg}]\npcc and p_value:{pcc}")
        return

    loss_f = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    start_epoch = cfg.START_EPOCH
    if cfg.RESUME:
        model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
        start_epoch = epoch
        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(start_epoch, cfg.MAX_EPOCH):
        # train for one epoch
        trainer.train(train_loader, model, loss_f, optimizer, epoch)
        # eval after training an epoch
        trainer.valid(valid_loader, model, loss_f, epoch)
        # update training lr every epoch
        scheduler.step()

        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - start_epoch, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), collector.best_valid_acc, collector.best_epoch)
    )


if __name__ == "__main__":
    args = parse_args()
    main(args, cfg)

