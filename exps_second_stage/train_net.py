import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from trainer import SpectrumTrainer
from trainer import MLPTrainer
from dpcv.modeling.networks.statistic_model import StatisticMLP
from dpcv.data.datasets.second_stage_dataset import SpectrumData, StatisticData
from mlflow import log_metric, log_param, log_params


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--bs", default=64, type=int, help="batch size in training")
    parser.add_argument("--max_epoch", default=3000, type=int, help="max training epochs")
    parser.add_argument("--lr_scale_rate", default=0.1, type=float, help="learning rate scale")
    parser.add_argument("--milestones", default=[1000, 1500], type=list, help="where to scale learning rate")
    parser.add_argument("--output_dir", default="result_static", type=str, help="where to save training output")
    args = parser.parse_args()
    return args


def main(test_only=None):
    log_param("exp", "swin")
    args = args_parse()
    log_params({"lr": args.lr, "epochs": args.max_epoch, "milestones": args.milestones, "bs": args.bs})

    dataset = {
        "train": "datasets/stage_two/swin_frame_pred_output/statistic_train_data.pkl",
        "valid": "datasets/stage_two/swin_frame_pred_output/statistic_valid_data.pkl",
        "test":  "datasets/stage_two/swin_frame_pred_output/statistic_test_data.pkl",
    }
    train_data_loader = DataLoader(
        StatisticData(dataset["train"]), batch_size=args.bs, shuffle=True,
        # num_workers=4,
        # StatisticData(dataset["train"]), batch_size = args.bs, shuffle = True
    )
    valid_data_loader = DataLoader(
        StatisticData(dataset["valid"]), batch_size=args.bs, shuffle=False,
        # num_workers=4,
        # StatisticData(dataset["valid"]), batch_size = args.bs, shuffle = False
    )
    test_data_loader = DataLoader(
        StatisticData(dataset["test"]), batch_size=args.bs, shuffle=False,
        # num_workers=4,
        # StatisticData(dataset["test"]), batch_size = args.bs
    )
    model = StatisticMLP().cuda()
    # model = StatisticConv1D().cuda()
    # model = SpectrumConv1D().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=args.lr_scale_rate, milestones=args.milestones)

    # trainer = SpectrumTrainer(max_epo=args.max_epoch, output_dir=args.output_dir)
    trainer = MLPTrainer(max_epo=args.max_epoch, output_dir=args.output_dir)

    if not test_only:
        for epo in range(args.max_epoch):
            trainer.train(model, train_data_loader, optimizer, epo)
            trainer.valid(model, valid_data_loader, epo)
            scheduler.step()
        acc = trainer.test(model, test_data_loader)
        trainer.save_model(model, epo, acc)
    else:
        checkpoint = torch.load(test_only)
        model.load_state_dict(checkpoint["model_state_dict"])
        acc = trainer.test(model, test_data_loader)


if __name__ == "__main__":
    os.chdir("..")
    main("result_static/checkpoint_2999.pkl")
