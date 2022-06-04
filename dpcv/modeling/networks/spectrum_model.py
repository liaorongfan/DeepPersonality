import torch.cuda
import torch.nn as nn
from dpcv.modeling.networks.build import NETWORK_REGISTRY
from dpcv import device
from dpcv.modeling.module.weight_init_helper import initialize_weights


class SpectrumConv1D(nn.Module):

    def __init__(self, channel=80, hidden_units=[128, 512, 2048]):
        super(SpectrumConv1D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 49), padding=(0, 24)
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(1, 25), padding=(0, 12)
            ),
            nn.ReLU(),
        )
        self.conv_up_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[1],
                kernel_size=(1, 9), padding=(0, 4),
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_units[1], out_channels=hidden_units[2],
                kernel_size=(1, 9), padding=(0, 4),
            ),
            nn.ReLU()
        )
        self.conv_down_scale = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_units[2], out_channels=hidden_units[1],
                kernel_size=(1, 9), padding=(0, 4),
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

    def __init__(self, signal_num=512, spectron_len=5, hidden_units=[256, 512, 1024, 512], init_weight=False):
        super(SpectrumConv1D2, self).__init__()
        # init_input
        self.init_input_conv2d = nn.Conv2d(
            in_channels=2, out_channels=spectron_len, kernel_size=(signal_num, 1), stride=1)

        # stage 1
        self.conv1d_up2_s2_1 = nn.Sequential(
            nn.Conv1d(in_channels=spectron_len, out_channels=hidden_units[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_units[0]),
            nn.LeakyReLU(),
        )  # (bs, 1024, 180）
        # self.shortcut_1 = nn.Sequential(
        #     nn.Conv1d(in_channels=spectron_len, out_channels=hidden_units[0], kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(),
        # )

        # stage 2
        self.con1d_stage = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(hidden_units[1]),
            nn.LeakyReLU(),
        )

        # stage 3
        self.conv1d_up2_s2_2 = nn.Sequential(
            # nn.Conv1d(
            #     in_channels=hidden_units[1], out_channels=hidden_units[1],
            #     kernel_size=3, padding=1, stride=1,
            # ),
            # nn.BatchNorm1d(hidden_units[1]),
            # nn.LeakyReLU(),

            # nn.Conv1d(
            #     in_channels=hidden_units[1], out_channels=hidden_units[1],
            #     kernel_size=3, padding=1, stride=1,
            # ),
            # nn.BatchNorm1d(hidden_units[1]),
            # nn.LeakyReLU(),

            nn.Conv1d(in_channels=hidden_units[1], out_channels=hidden_units[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_units[2]),
            nn.LeakyReLU(),
        )  # (bs, 2048, 90)
        self.shortcut_2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units[1], out_channels=hidden_units[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        # stage 4
        self.conv1d_s2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units[2], out_channels=hidden_units[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_units[2]),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=hidden_units[2], out_channels=hidden_units[2], kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_units[2]),
            nn.LeakyReLU(),
        )

        # regressor
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            # nn.Linear(hidden_units[2], hidden_units[2]),
            # nn.LeakyReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_units[2], hidden_units[3]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_units[3], 5),
        )

        if init_weight:
            print("init weights")
            for m in self.modules():
                if isinstance(m, nn.Sequential):
                    for m_i in m.modules():
                        if isinstance(m_i, nn.Conv1d):
                            nn.init.kaiming_normal_(m_i.weight)
                        elif isinstance(m_i, nn.BatchNorm1d):
                            nn.init.constant_(m_i.weight, 1)
                            nn.init.constant_(m_i.bias, 0)
                        elif isinstance(m_i, nn.Conv2d):
                            nn.init.kaiming_normal_(m_i.weight)

                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # init input
        x = self.init_input_conv2d(x).squeeze(dim=2)
        # stage 1:
        # x_1 = x
        x = self.conv1d_up2_s2_1(x)
        # x_shortcut_1 = self.shortcut_1(x_1)
        # x += x_shortcut_1
        # stage 2:
        x = self.con1d_stage(x)
        # stage 3:
        x_3 = x
        x = self.conv1d_up2_s2_2(x)
        x_shortcut_3 = self.shortcut_2(x_3)
        x += x_shortcut_3
        # stage 4:
        x = self.conv1d_s2(x)
        # regressor
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


class SpectrumFeatConv1D(nn.Module):

    def __init__(self, channel=80, signal_num=512, hidden_units=[64, 256, 1024], initialize=False):
        super(SpectrumFeatConv1D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=hidden_units[0], kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=(1, 5), padding=(0, 2)
            ),
            nn.ReLU(),
        )

        self.conv_up_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units[1], out_channels=hidden_units[1], kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units[1], out_channels=hidden_units[2], kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU()
        )

        self.conv_down_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units[2], out_channels=hidden_units[1], kernel_size=(1, 3), padding=(0, 1),
            ),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units[1],  out_channels=1, kernel_size=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(1, channel)
            ),
        )

        self.trait_fc = nn.Linear(signal_num, 5)
        if initialize:
            initialize_weights(self)

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


def conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 25), stride=1, padding=(0, 12), bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 25), stride=(1, 2*stride), padding=(0, 12), bias=False)
    else:
        raise ValueError("wrong stride value")


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(1, 2 * stride), bias=False)
    else:
        raise ValueError("wrong stride value")


class InitStage(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super(InitStage, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 49), stride=(1, 1), padding=(0, 24), bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4))

    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        return x


class BasicBlock(nn.Module):
    """
    build  conv block for resnet18 architecture
    """
    expansion = 1

    def __init__(self, conv_type, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_type(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FeatResNet(nn.Module):

    def __init__(
        self, in_channels, init_stage, block, conv,
        channels=[64, 128, 256, 512],  # default resnet stage channel settings
        layers=[2, 2, 2, 2],  # default resnet18 layers setting
        out_spatial=(512, 1),
        zero_init_residual=False
    ):
        super(FeatResNet, self).__init__()
        assert len(conv) == 2, "conv should be a list containing <conv3x3 conv1x1> or <conv1x9, conv1x1> function"

        self.inplanes = channels[0]
        self.conv_3x3 = conv[0]
        self.conv_1x1 = conv[1]
        self.init_stage = init_stage(in_channels, channels[0])
        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(out_spatial)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,  block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv_1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.conv_3x3, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.conv_3x3, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_stage(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x_out = x_avg.view(x_avg.size(0), -1)
        return x


class FeatResNet18(nn.Module):

    def __init__(self, channels=[64, 128, 256, 512], signal_num=512):
        super(FeatResNet18, self).__init__()
        self.main_branch = FeatResNet(
            in_channels=2, init_stage=InitStage,
            block=BasicBlock, conv=[conv1x9, conv1x1],
            channels=channels,
            out_spatial=(signal_num, 1)
        )
        out_unit = channels[-1] * signal_num

        self.linear1 = nn.Linear(out_unit, 512)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 5)

    def forward(self, feat_input):
        x = self.main_branch(feat_input)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
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
    return SpectrumConv1D2(signal_num=cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)


@NETWORK_REGISTRY.register()
def spectrum_Feat_conv_model(cfg):
    # return SpectrumConv1D().to(device=torch.device("gpu" if torch.cuda.is_available() else "cpu"))
    # sample_channel = 100
    return SpectrumFeatConv1D(channel=cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)


@NETWORK_REGISTRY.register()
def spectrum_feat_resnet18(cfg=None):
    return FeatResNet18(signal_num=cfg.MODEL.SPECTRUM_CHANNEL).to(device=device)


if __name__ == "__main__":
    # x = torch.randn((1, 2, 512, 80)).cuda()
    # model = spectrum_feat_resnet18()
    # y = model(x)
    # print(y)
    # inputs = torch.randn(20, 16, 50)
    # m = nn.Conv1d(16, 2, 3, stride=1)
    # output = m(inputs)
    # print(output.shape)
    # target output size of 5
    # m = nn.AdaptiveAvgPool1d(1)
    # input = torch.randn(1, 64, 8)
    # output = m(input)
    # print(output.shape)
    net = SpectrumConv1D2()
    x = torch.randn((1, 2, 512, 180))
    y = net(x)
    print(y.shape)
    import torchvision.models as models
    net = models.GoogLeNet