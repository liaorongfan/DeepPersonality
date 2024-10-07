import torch
import torch.nn as nn
from dpcv.modeling.module.resnet_tv import Bottleneck, BasicBlock, conv1x1, conv3x3
from dpcv.modeling.module.resnet_tv import model_zoo, model_urls
from .build import NETWORK_REGISTRY


class SoundNet(nn.Module):

    def __init__(self, input_dim, out_dim=5, scale=2, use_sigmoid=True):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(512 * scale)),
            nn.BatchNorm1d(int(512 * scale)),
            nn.ReLU(),
            nn.Linear(int(512 * scale), int(128 * scale)),
            nn.BatchNorm1d(int(128 * scale)),
            nn.ReLU(),
            nn.Linear(int(128 * scale), out_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        # x = x.reshape(-1, self.input_dim)
        f = self.fc[:-2](x)
        x = self.fc[-2:](f)

        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


@NETWORK_REGISTRY.register()
def sound_model(cfg):
    input_dim = cfg.MODEL.INPUT_DIM
    model = SoundNet(input_dim, out_dim=cfg.MODEL.NUM_CLASS, use_sigmoid=True)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
