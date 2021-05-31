import torch
import torch.nn as nn


def conv_block(in_channel, out_channel):

    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
    return layer


def make_layers(in_channel):
    layers = []
    for channels in [16 * (2 ** x) for x in range(8)]:
        conv_layer = conv_block(in_channel, channels)
        layers.append(conv_layer)
        in_channel = channels
    return nn.Sequential(*layers)
    

class PortraitNet(nn.Module):

    def __init__(self, input_dimension=1):
        super(PortraitNet, self).__init__()
        self.features = make_layers(input_dimension)
        self.dropout = nn.Dropout(0.8)
        self.regressor = nn.Sequential(
            nn.Linear(2048, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        y = self.regressor(x)
        return y


def get_portrait_model():
    model = PortraitNet()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


if __name__ == "__main__":
    inputs = torch.rand((4, 1, 208, 208))
    net = PortraitNet()
    print(net)
    output = net(inputs)
    print(output.shape)
