import torch
import torch.nn as nn
from dpcv.modeling.module.weight_init_helper import initialize_weights
from .build import NETWORK_REGISTRY


class AudioInterpretNet(nn.Module):

    def __init__(self, init_weights=True):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool1d(10),
            nn.Dropout(0.1),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool1d(8),
            nn.Dropout(0.5),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=6, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool1d(8),
            nn.Dropout(0.5),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 5)

        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def get_model(cfg, pretrained=False, **kwargs):
    model = AudioInterpretNet()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@NETWORK_REGISTRY.register()
def interpret_audio_model(cfg):
    model = AudioInterpretNet()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


if __name__ == "__main__":
    dumy = torch.randn((2, 1, 30604))
    model = AudioInterpretNet()
    out = model(dumy)
    print(out.shape)
