import torch.nn as nn
from .build import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class SpectrumConv1D(nn.Module):

    def __init__(self):
        super(SpectrumConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(1, 7))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1, 5))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=(1, 3))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64,  out_channels=1, kernel_size=(1, 1))
        self.adp = nn.AdaptiveAvgPool1d(1)
        self.fcn = nn.Linear(80, 5)

    def forward(self, x):
        x = x[..., None]
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.squeeze(1)
        x = self.adp(x)
        x = x.squeeze()
        x = self.fcn(x)
        return x
