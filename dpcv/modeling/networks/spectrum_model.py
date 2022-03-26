import torch.cuda
import torch.nn as nn
from .build import NETWORK_REGISTRY
from dpcv import device


class SpectrumConv1D(nn.Module):

    def __init__(self, channel=50, hidden_units=[64, 256, 1024]):
        super(SpectrumConv1D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(1, 5), padding=(0, 2)
            ),
            nn.ReLU(),
        )
        self.conv_up_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU()
        )
        self.conv_down_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1],  out_channels=1, kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=(1, channel)
            ),
        )

    def forward(self, x):
        x = self.conv_in(x)           # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        identical = x
        x = self.conv_up_scale(x)     # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.conv_down_scale(x)   # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x += identical
        x = self.conv_out(x)          # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze(1)              # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze()               # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        return x


class SpectrumConv1D2(nn.Module):

    def __init__(self, channel=50, hidden_units=[64, 256, 1024, 2048]):
        super(SpectrumConv1D2, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(1, 5), padding=(0, 2)
            ),
            nn.ReLU(),
        )

        self.conv_up_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU()
        )
        self.conv_stage = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )
        self.conv_down_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1],  out_channels=1, kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=(1, channel)
            ),
        )

    def forward(self, x):
        x = self.conv_in(x)           # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        # feature_in = x
        x = self.conv_up_scale(x)
        # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        # identical = x
        x = self.conv_stage(x)
        # x += identical

        x = self.conv_down_scale(x)   # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        # x += feature_in

        x = self.conv_out(x)          # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze(1)              # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze()               # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        return x


class SpectrumFeatConv1D(nn.Module):

    def __init__(self, channel=50, signal_num=512, hidden_units=[64, 256, 1024]):
        super(SpectrumFeatConv1D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(1, 5), padding=(0, 2)
            ),
            nn.ReLU(),
        )
        self.conv_up_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[2],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU()
        )
        self.conv_down_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[1],
                kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1],  out_channels=1, kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=(1, channel)
            ),
        )

        self.trait_fc = nn.Linear(signal_num, 5)

    def forward(self, x):
        x = self.conv_in(x)           # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        identical = x
        x = self.conv_up_scale(x)     # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.conv_down_scale(x)   # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x += identical
        x = self.conv_out(x)          # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze(1)              # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = x.squeeze()               # (bs, 2, 5, 50) --> (bs, 64, 5, 50)
        x = self.trait_fc(x)
        return x



@NETWORK_REGISTRY.register()
def spectrum_conv_model(cfg):
    # return SpectrumConv1D().to(device=torch.device("gpu" if torch.cuda.is_available() else "cpu"))
    # sample_channel = 100
    return SpectrumConv1D(cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)


@NETWORK_REGISTRY.register()
def spectrum_conv_model2(cfg):
    # return SpectrumConv1D().to(device=torch.device("gpu" if torch.cuda.is_available() else "cpu"))
    # sample_channel = 100
    return SpectrumConv1D2(cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)


@NETWORK_REGISTRY.register()
def spectrum_Feat_conv_model(cfg):
    # return SpectrumConv1D().to(device=torch.device("gpu" if torch.cuda.is_available() else "cpu"))
    # sample_channel = 100
    return SpectrumFeatConv1D(cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)