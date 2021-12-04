import torch
import torch.nn as nn
from .build import LOSS_FUNC_REGISTRY


def one_hot_CELoss(pred, label):
    bs = label.size(0)
    log_prob = torch.log_softmax(pred, dim=1)
    loss = -torch.sum(log_prob * label) / bs
    return loss


class BellLoss:
    def __init__(self, gama=300, theta=9):
        self.gama = torch.as_tensor(gama)
        self.theta = torch.as_tensor(theta)

    def __call__(self, pred, label):
        exponent = - torch.square(pred - label) / (2 * torch.square(self.theta))
        loss = self.gama * (1 - torch.exp(exponent)).sum()
        return loss


@LOSS_FUNC_REGISTRY.register()
def crnet_loss_func():
    return {
        "ce_loss": one_hot_CELoss,
        "bell_loss": BellLoss(),
        "mse_loss": nn.MSELoss(),
        "l1_loss": nn.L1Loss()
    }


if __name__ == "__main__":
    pre = torch.Tensor([[[0.1, 0.2, 0.3, 2],
                         [0.3, 0.5, 0.4, 4]],

                        [[0.1, 0.2, 0.3, 1],
                         [0.3, 0.5, 0.4, 3]]])
    print("pre:", pre.shape)
    oh_label = torch.tensor([[[0, 1, 0, 0],
                              [0, 0, 1, 0]],
                            [[1, 0, 0, 0],
                             [0, 1, 0, 0]]])
    print("oh_label:", oh_label.shape)
    loss = one_hot_CELoss(pre, oh_label)
    print(loss)

    pred = torch.randn(2, 5)
    label = torch.randn(2, 5)
    bell_loss = BellLoss()
    b_loss = bell_loss(pred, label)
    print(b_loss)
