import os
import torch
import random
import numpy as np
import argparse
from dpcv.tools.draw import plot_line
# import psutil


def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


def parse_args():
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
    return args


class TrainSummary:
    def __init__(self):
        self.iter_loss_record = {"train": [], "valid": []}
        self.iter_acc_record = {"train": [], "valid": []}
        self.epoch_loss = {"train": [], "valid": []}
        self.epoch_acc = {"train": [], "valid": []}
        self._best_acc = [0]
        self._best_epoch = [0]
        self.model_save_flag = []
        self.valid_info = {"ocean_acc": []}

    def update_best_acc(self, acc):
        self._best_acc.append(acc)

    def update_best_epoch(self, epo):
        self._best_epoch.append(epo)

    def update_model_save_flag(self, flag):
        """
        mark whether to save model weights
        args:
            flag (int 0 / 1): 0, mark as not save, otherwise save

        """
        self.model_save_flag.append(flag)

    def record_valid_ocean_acc(self, ocean):
        self.valid_info["ocean_acc"].append(ocean)

    def record_train_loss(self, loss_train):
        if isinstance(loss_train, float):
            self.iter_loss_record["train"].append(loss_train)
        elif isinstance(loss_train, list):
            self.iter_loss_record["train"].extend(loss_train)
        else:
            raise ValueError("loss should be a float number or a list of float number")
        self.update_epoch_train_loss(loss_train)

    def record_train_acc(self, acc_train):
        if isinstance(acc_train, float):
            self.iter_acc_record["train"].append(acc_train)
        elif isinstance(acc_train, list):
            self.iter_acc_record["train"].extend(acc_train)
        else:
            raise ValueError("acc should be a float number or a list of float number")
        self.update_epoch_train_acc(acc_train)

    def record_valid_loss(self, loss_valid):
        if isinstance(loss_valid, float):
            self.iter_loss_record["valid"].append(loss_valid)
        elif isinstance(loss_valid, list):
            self.iter_loss_record["valid"].extend(loss_valid)
        else:
            raise ValueError("loss should be a float number or a list of float number")
        self.update_epoch_valid_loss(loss_valid)

    def record_valid_acc(self, acc_valid):
        if isinstance(acc_valid, float):
            self.iter_acc_record["valid"].append(acc_valid)
        elif isinstance(acc_valid, list):
            self.iter_acc_record["valid"].extend(acc_valid)
        else:
            raise ValueError("acc should be a float number or a list of float number")
        self.update_epoch_valid_acc(acc_valid)

    @property
    def model_save(self):
        return self.model_save_flag[-1] > 0

    @property
    def best_acc(self):
        return self._best_acc[-1]

    @property
    def best_epoch(self):
        return self._best_epoch[-1]

    @property
    def mean_train_acc(self):
        return np.mean(np.array(self.iter_acc_record["train"]))

    @property
    def mean_valid_acc(self):
        return np.mean(np.array(self.iter_acc_record["valid"]))

    @property
    def mean_train_loss(self):
        return np.mean(np.array(self.iter_loss_record["train"]))

    @property
    def mean_valid_loss(self):
        return np.mean(np.array(self.iter_loss_record["valid"]))

    @property
    def valid_ocean_acc(self):
        return self.valid_info["ocean_acc"][-1]

    def update_epoch_train_loss(self, loss_list):
        """record mean loss value of each training epoch"""
        self.epoch_loss["train"].append(np.mean(np.array(loss_list)))

    def update_epoch_train_acc(self, acc_list):
        """record mean loss value of each training epoch"""
        self.epoch_acc["train"].append(np.mean(np.array(acc_list)))

    def update_epoch_valid_loss(self, loss_list):
        """record mean loss value of each validation epoch"""
        self.epoch_loss["valid"].append(np.mean(np.array(loss_list)))

    def update_epoch_valid_acc(self, acc_list):
        """record mean loss value of each validation epoch"""
        self.epoch_acc["valid"].append(np.mean(np.array(acc_list)))

    def draw_epo_info(self, epochs, log_dir):
        plt_x = np.arange(0, epochs + 1)
        plot_line(
            plt_x, self.epoch_loss["train"],
            plt_x, self.epoch_loss["valid"],
            mode="loss", out_dir=log_dir
        )
        plot_line(
            plt_x, self.epoch_acc["train"],
            plt_x, self.epoch_acc["valid"],
            mode="acc", out_dir=log_dir
        )

    def draw_batch_info(self, log_dir):
        plt_x_batch = np.arange(1, len(self.iter_loss_record["train"]) + 1)
        plot_line(
            plt_x_batch, self.iter_loss_record["train"],
            plt_x_batch, self.iter_acc_record["train"],
            mode="batch info", out_dir=log_dir
        )
