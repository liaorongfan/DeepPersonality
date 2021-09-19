import torch
import torch.nn as nn


class AudioInterpretNet(nn.Module):

    def __init__(self):
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

        self.fc = nn.Linear(2000, 5)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.fc(x)
        return x

