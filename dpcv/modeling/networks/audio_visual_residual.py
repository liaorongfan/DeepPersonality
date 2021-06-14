import torch
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
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=stride, padding=(0, 4), bias=False)


def aud_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(1, 2*stride), bias=False)


class VisInitStage(nn.Module):
    def __init__(self):
        super(VisInitStage, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class AudInitStage(nn.Module):
    def __init__(self):
        super(AudInitStage, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 49), stride=(1, 4), padding=(0, 24), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4))

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class VisBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(VisBasicBlock, self).__init__()
        self.conv1 = vis_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = vis_conv3x3(planes, planes)
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


class AudBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AudBasicBlock, self).__init__()
        self.conv1 = aud_conv1x9(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = aud_conv1x9(planes, planes)
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


class ResNet18(nn.Module):

    def __init__(self, init_stage, block, zero_init_residual=False):
        super(ResNet18, self).__init__()
        self.inplanes = 32
        self.init_stage = init_stage()
        self.layer1 = self._make_layer(block, 32, 2)
        self.layer2 = self._make_layer(block, 64, 2, stride=2)
        self.layer3 = self._make_layer(block, 128, 2, stride=2)
        self.layer4 = self._make_layer(block, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,  block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                vis_conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_stage(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class AudioVisualResNet18(nn.Module):
    def __init__(self):
        super(AudioVisualResNet18, self).__init__()
        self.audio_branch = ResNet18(AudInitStage, AudBasicBlock)
        self.visual_branch = ResNet18(VisInitStage, VisBasicBlock)
        self.linear = nn.Linear(512, 5)

    def forward(self, aud_input, vis_input):
        aud_x = self.audio_branch(aud_input)
        vis_x = self.visual_branch(vis_input)
        x = torch.cat([aud_x, vis_x], dim=-1)
        x = self.linear(x)
        x = torch.tanh(x)
        return x


if __name__ == "__main__":
    aud = torch.randn(2, 1, 1, 50176)
    vis = torch.randn(2, 3, 224, 224)
    multi_model = AudioVisualResNet18()
    y = multi_model(aud, vis)
    print(y)

