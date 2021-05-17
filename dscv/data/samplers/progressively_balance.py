"""
# @file name  : progressively_balance.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2021-02-28
# @brief      : 渐进式平衡采样，2020-ICLR-Decoupling Representation and Classifier
"""
# import matplotlib
# matplotlib.use('agg')
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, '..'))
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets.cifar_longtail import CifarLTDataset
from tools.common_tools import check_data_dir
from config.cifar_config import cfg
import matplotlib.pyplot as plt


class ProgressiveSampler(object):
    def __init__(self, dataset, max_epoch):
        self.max_epoch = max_epoch
        self.dataset = dataset      # dataset
        self.train_targets = [int(info[1]) for info in dataset.img_info]    #  hard code，记录各个样本的标签
        self.nums_per_cls = dataset.nums_per_cls        # 记录每个类别的样本数量

    def _cal_class_prob(self, q):
        """
        根据q值计算每个类的采样概率，公式中的 p_j
        :param q: float , [0, 1]
        :return: list,
        """
        num_pow = list(map(lambda x: pow(x, q), self.nums_per_cls))
        sigma_num_pow = sum(num_pow)
        cls_prob = list(map(lambda x: x/sigma_num_pow, num_pow))
        return cls_prob

    def _cal_pb_prob(self, t):
        """
        progressively-balanced 概率计算
        :param t: 当前epoch数
        :return:
        """
        p_ib = self._cal_class_prob(q=1)
        p_cb = self._cal_class_prob(q=0)
        p_pb = (1 - t/self.max_epoch) * np.array(p_ib) + (t/self.max_epoch) * np.array(p_cb)

        p_pb /= np.array(self.nums_per_cls)  # very important！由于pytorch的sampler机制是按每个样本采样，所以要除以样本总数
        return p_pb.tolist()

    def __call__(self, epoch):
        """
        生成sampler
        :param epoch:
        :return:
        """
        p_pb = self._cal_pb_prob(t=epoch)
        p_pb = torch.tensor(p_pb, dtype=torch.float)
        samples_weights = p_pb[self.train_targets]  # 计算每个样本被采样的权重，这里是依据样本的类别来赋权，self.train_targets是标签
        # weights：要求是每个样本赋予weight
        # num_samples：该sampler一个epoch采样数量
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights))
        # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=1000)
        return sampler, p_pb

    def plot_line(self):
        for i in range(self.max_epoch):
            _, weights = self(i)
            if i % 20 == 19:
                x = range(len(weights))
                plt.plot(x, weights, label="t="+str(i))
        plt.legend()
        plt.title("max epoch="+str(self.max_epoch))
        plt.xlabel("class index sorted by numbers")
        plt.ylabel("weights")
        plt.show()


if __name__ == '__main__':
    # 设置路径
    train_dir = r"G:\deep_learning_data\cifar10\cifar10_train"
    check_data_dir(train_dir)
    train_data = CifarLTDataset(root_dir=train_dir, transform=cfg.transforms_train, isTrain=True)

    max_epoch = 200
    sampler_generator = ProgressiveSampler(train_data, max_epoch)
    sampler_generator.plot_line()

    for epoch in range(max_epoch):
        if epoch % 20 != 19:
            continue

        sampler, _ = sampler_generator(epoch)
        train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False, num_workers=cfg.workers,
                                  sampler=sampler)
        labels = []
        for data in train_loader:
            _, label, _ = data
            labels.extend(label.tolist())
        print("Epoch:{}, Counter:{}".format(epoch, Counter(labels)))



