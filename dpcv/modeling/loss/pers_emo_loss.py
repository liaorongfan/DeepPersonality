import torch
import torch.nn.functional as F
from .build import LOSS_FUNC_REGISTRY


def per_emo_loss(p_score, p_label, e_score, e_label, p_co, e_co, x_ep, use_adv=False):
    p_mark = torch.cat([torch.ones((p_co.shape[0], 1)), torch.zeros((p_co.shape[0], 1))], 1).cuda()
    e_mark = torch.cat([torch.zeros((e_co.shape[0], 1)), torch.ones((e_co.shape[0], 1))], 1).cuda()
    l_p = F.smooth_l1_loss(p_score, p_label) / 10  # equal to aggregation output of personality prediction
    l_e = F.smooth_l1_loss(e_score, e_label)
    l_ep = F.smooth_l1_loss(x_ep, p_label)

    l_d_p = F.binary_cross_entropy(p_co, p_mark)
    l_d_e = F.binary_cross_entropy(e_co, e_mark)
    l_d = l_d_p + l_d_e
    if not use_adv:
        return l_p + l_e + 0.1 * l_ep + 0.1 * l_d

    l_d_p_adv = F.binary_cross_entropy(p_co, e_mark)
    l_d_e_adv = F.binary_cross_entropy(e_co, p_mark)
    l_adv = l_d_p + l_d_e + l_d_p_adv + l_d_e_adv

    return l_p + l_e + 0.1 * l_ep + 0.1 * l_d + 0.1 * l_adv


@LOSS_FUNC_REGISTRY.register()
class PeremonLoss:
    def __call__(self, p_score, p_label, e_score, e_label, p_co, e_co, x_ep, use_adv=False):
        p_mark = torch.cat([torch.ones((p_co.shape[0], 1)), torch.zeros((p_co.shape[0], 1))], 1).cuda()
        e_mark = torch.cat([torch.zeros((e_co.shape[0], 1)), torch.ones((e_co.shape[0], 1))], 1).cuda()
        l_p = F.smooth_l1_loss(p_score, p_label) / 10  # equal to aggregation output of personality prediction
        l_e = F.smooth_l1_loss(e_score, e_label)
        l_ep = F.smooth_l1_loss(x_ep, p_label)

        l_d_p = F.binary_cross_entropy(p_co, p_mark)
        l_d_e = F.binary_cross_entropy(e_co, e_mark)
        l_d = l_d_p + l_d_e
        if not use_adv:
            return l_p + l_e + 0.1 * l_ep + 0.1 * l_d

        l_d_p_adv = F.binary_cross_entropy(p_co, e_mark)
        l_d_e_adv = F.binary_cross_entropy(e_co, p_mark)
        l_adv = l_d_p + l_d_e + l_d_p_adv + l_d_e_adv

        return l_p + l_e + 0.1 * l_ep + 0.1 * l_d + 0.1 * l_adv