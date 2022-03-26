"""
code modified from https://github.com/clcarwin/sphereface_pytorch.git
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from dpcv.modeling.module.weight_init_helper import initialize_weights
from .build import NETWORK_REGISTRY


class SphereFEM(nn.Module):
    def __init__(self, pre_trained=None):
        super(SphereFEM, self).__init__()
        self.pre_trained = pre_trained
        # input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 7, 512)

        if pre_trained:
            self.load_pre_trained_model()

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return x

    def load_pre_trained_model(self):
        pass


class PersEmoN(nn.Module):
    def __init__(self, feature_extractor, init_weights=True, return_feature=False):
        super(PersEmoN, self).__init__()
        self.return_feature = return_feature
        self.efm = feature_extractor
        self.pam = nn.Linear(512, 5)
        self.eam = nn.Linear(512, 2)
        self.ram = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
        self.data_classifier = nn.Linear(512, 2)

        if init_weights:
            initialize_weights(self)

    def forward(self, x_p, x_e):
        x_p = self.efm(x_p)
        p_coherence = F.softmax(self.data_classifier(x_p), 1)
        p_score = self.pam(x_p)
        p_score = torch.sigmoid(p_score)

        x_e = self.efm(x_e)
        e_coherence = F.softmax(self.data_classifier(x_e), 1)
        e_score = self.eam(x_e)
        x_ep = self.ram(e_score)
        e_score = torch.tanh(e_score)

        if self.return_feature:
            return p_score, p_coherence, e_score, e_coherence, x_ep, x_p

        return p_score, p_coherence, e_score, e_coherence, x_ep


def get_pers_emo_model():
    multi_modal_model = PersEmoN(SphereFEM())
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


@NETWORK_REGISTRY.register()
def pers_emo_model(cfg=None):
    multi_modal_model = PersEmoN(SphereFEM(), return_feature=cfg.MODEL.RETURN_FEATURE)
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


if __name__ == "__main__":
    fem = PersEmoN(SphereFEM())
    inputs_p = torch.randn((100, 3, 112, 112))
    inputs_e = torch.randn((100, 3, 112, 112))
    out = fem(inputs_p, inputs_e)
    for item in out:
        print(item.shape)
    # print(out[1])
