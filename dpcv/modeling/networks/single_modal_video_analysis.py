import torch
import torch.nn as nn


class SequenceBasedModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 5, 5)),
            nn.BatchNorm3d(100),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.block_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 5, 5)),
            nn.BatchNorm3d(100),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.block_3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 5, 5)),
            nn.BatchNorm3d(100),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.block_4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 5, 5)),
            nn.BatchNorm3d(100),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.block_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc_out = nn.Linear(512, 5)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.fc_out(x)
        return x



