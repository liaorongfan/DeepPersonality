import torch
import torch.nn as nn
from dpcv.modeling.module.resnet_tv import Bottleneck, BasicBlock, conv1x1, conv3x3
from dpcv.modeling.module.resnet_tv import model_zoo, model_urls
from .build import NETWORK_REGISTRY


class MetaFCNet(nn.Module):

    def __init__(self, input_dim, out_dim=5,  use_sigmoid=False):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        f = self.fc[:-2](x)
        x = self.fc[-2:](f)

        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


@NETWORK_REGISTRY.register()
def meta_fusing_model(cfg):
    input_dim = cfg.MODEL.INPUT_DIM
    use_sigmoid = False
    if cfg.DATA.AU is not None:
        use_sigmoid = True
    model = MetaFCNet(input_dim, out_dim=cfg.MODEL.NUM_CLASS, use_sigmoid=use_sigmoid)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
