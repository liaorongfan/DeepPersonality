# import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# consider add pre-train weight
# model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


def vis_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def vis_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def aud_conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=(1, 2*stride), padding=(0, 4), bias=False)
    else:
        raise ValueError("wrong stride value")


def aud_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    if stride == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    elif stride == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(1, 2 * stride), bias=False)
    else:
        raise ValueError("wrong stride value")


class VisInitStage(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(VisInitStage, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class AudInitStage(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(AudInitStage, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 49), stride=(1, 4), padding=(0, 24), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4))

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class BiModalBasicBlock(nn.Module):
    """
    build visual and audio conv block for resnet18 architecture
    """
    expansion = 1

    def __init__(self, conv_type, inplanes, planes, stride=1, downsample=None):
        super(BiModalBasicBlock, self).__init__()
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


class AudioVisualResNet(nn.Module):

    def __init__(self, in_channels, init_stage, block, conv,
                 channels=[64, 128, 256, 512],  # default resnet stage channel settings
                 layers=[2, 2, 2, 2],  # default resnet18 layers setting
                 out_spatial=(1, 1),
                 zero_init_residual=False):
        super(AudioVisualResNet, self).__init__()

        assert init_stage.__name__ in ["AudInitStage", "VisInitStage"], \
            "init conv stage should be 'AudInitStage' or 'VisInitStage'"
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
