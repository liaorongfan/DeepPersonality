import torch
import torch.nn as nn
from .build import NETWORK_REGISTRY


class StatisticMLP(nn.Module):
    def __init__(self, hidden_units=256):
        super(StatisticMLP, self).__init__()
        # log_param("hidden_units", hidden_units)
        self.input_layer = nn.Linear(12 * 5, hidden_units)
        self.relu_1 = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(hidden_units, hidden_units)
        self.relu_2 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_units, int(hidden_units / 4))
        self.relu_3 = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(int(hidden_units / 4), 5)
        # log_param("model", "MLP")

    def forward(self, x):
        x = x.reshape(-1, 60)
        x = self.input_layer(x)
        x = self.relu_1(x)
        x = self.hidden_layer_1(x)
        x = self.relu_2(x)
        x = self.hidden_layer_2(x)
        x = self.relu_3(x)
        # x = self.dropout(x)
        x = self.output_layer(x)
        return x


class StatisticConv1D(nn.Module):

    def __init__(self):
        super(StatisticConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(1, 12))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        # log_param("model", "Conv1d:conv4")

    def forward(self, x):
        x = x[..., None]
        x = x.permute(0, 3, 2, 1)   # (bs, 1, 5, 12)
        x = self.conv1(x)           # (bs, 64, 5, 1)
        x = self.relu1(x)
        x = self.conv2(x)           # (bs, 128, 5, 1)
        x = self.relu2(x)
        x = self.conv3(x)           # (bs, 1, 5, 1)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.squeeze(1).squeeze()  # (bs, 5)
        return x


@NETWORK_REGISTRY.register()
def statistic_mlp(cfg):
    model = StatisticMLP()
    return model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
