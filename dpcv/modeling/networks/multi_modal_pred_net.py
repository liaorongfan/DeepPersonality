import torch
import torch.nn as nn
from dpcv.modeling.module.resnet_tv import Bottleneck, BasicBlock, conv1x1, conv3x3
from dpcv.modeling.module.resnet_tv import model_zoo, model_urls
from .build import NETWORK_REGISTRY


class ResNetFeatureExtractor(nn.Module):
    """
    Note: that class is not a formal resnet but with a sigmoid function for the last fc layer
    """
    def __init__(
            self, block, layers, num_classes=1000,
            init_weights=True, zero_init_residual=False, sigmoid_output=True,
            return_feat=False,
    ):
        super(ResNetFeatureExtractor, self).__init__()
        self.return_feature = return_feat
        self.sigmoid_output = sigmoid_output
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool14x14 = nn.MaxPool2d(kernel_size=14)
        self.maxpool7x7 = nn.MaxPool2d(kernel_size=7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # if init_weights:
        #     initialize_weights(self)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x_3 = self.layer3(x)
        x_3_1 = self.maxpool14x14(x_3).reshape(-1, 1024)
        x_3_2 = self.avgpool(x_3).reshape(-1, 1024)

        x_4 = self.layer4(x_3)
        x_4_1 = self.maxpool7x7(x_4).reshape(-1, 2048)
        x_4_2 = self.avgpool(x_4).reshape(-1, 2048)

        feat = torch.cat([x_3_1, x_3_2, x_4_1, x_4_2], dim=1)
        return feat


def resnet101_visual_feature_extractor(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNetFeatureExtractor(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class VisualFCNet(nn.Module):

    def __init__(self, input_dim, out_dim=5, use_sigmoid=True):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = x.mean(dim=1)
        if self.use_sigmoid:
            return self.sigmoid(x)
        return x


class AudioFCNet(nn.Module):

    def __init__(self, input_dim, out_dim=5, spectrum_channel=15, use_sigmoid=True):
        super().__init__()
        self.spectrum_channel = spectrum_channel
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        x = x.view(-1, self.spectrum_channel * 128)
        x = (x - x.mean()) / x.std()
        x = self.fc(x)
        if self.use_sigmoid:
            return self.sigmoid(x)
        return x


@NETWORK_REGISTRY.register()
def multi_modal_visual_model(cfg):
    model = VisualFCNet(6144)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@NETWORK_REGISTRY.register()
def multi_modal_audio_model(cfg):
    if cfg.DATA.SESSION in ["talk", "animal", "ghost", "lego"]:
        dim = cfg.MODEL.SPECTRUM_CHANNEL * 128
        use_sigmoid = False
    else:
        dim = 15 * 128
        use_sigmoid = True
    model = AudioFCNet(dim, spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL, use_sigmoid=use_sigmoid)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


if __name__ == "__main__":
    model = resnet101_visual_feature_extractor()
    fake = torch.randn(6, 3, 224, 224)
    output = model(fake)
    print(output.shape)
