import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import NETWORK_REGISTRY
from dpcv.modeling.module.weight_init_helper import initialize_weights
# import torch.utils.model_zoo as model_zoo

backbone = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class DAN(nn.Module):

    def __init__(self, features, num_classes=5, init_weights=True, return_feature=False):
        super(DAN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.AdaptiveMaxPool2d((7, 7))
        self.linear_1 = nn.Linear(50176, 1024)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(1024, num_classes)
        self.return_feature = return_feature
        if init_weights:
            # self._initialize_weights()
            initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x1 = self.avgpool(x)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = self.maxpool(x)
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        feat = self.leaky_relu(x)
        x = self.dropout(feat)  # add dropout to enhance generalization ability
        x = self.linear_2(x)  # add another linear lay and activation function to enhance nonlinear mapping
        x = torch.sigmoid(x)
        if self.return_feature:
            return x, feat
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AudLinearRegressor(nn.Module):

    def __init__(self, input_units=79534):
        super(AudLinearRegressor, self).__init__()
        self.linear = nn.Linear(input_units, 5)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1).squeeze(1)
        return x


@NETWORK_REGISTRY.register()
def get_aud_linear_regressor(cfg=None):
    model = AudLinearRegressor()
    model.to(device=(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    return model


@NETWORK_REGISTRY.register()
def get_true_personality_aud_linear_regressor(cfg=None):
    model = AudLinearRegressor(input_units=244832)
    model.to(device=(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    return model


@NETWORK_REGISTRY.register()
def dan_model(cfg):

    kwargs = {'init_weights': True}
    kwargs["return_feature"] = cfg.MODEL.RETURN_FEATURE
    if cfg.MODEL.PRETRAIN:
        kwargs["init_weights"] = False

    dan = DAN(make_layers(backbone['VGG16'], batch_norm=True), **kwargs)

    if cfg.MODEL.PRETRAIN:
        print("load pretained model weights")
        pretrained_dict = torch.load("../pre_trained_weights/vgg16_bn-6c64b313.pth")
        model_dict = dan.state_dict()
        # 1. filter out unnecessary keys -------------------------------------------------------------------------------
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict --------------------------------------------------------------
        model_dict.update(pretrained_dict)
        dan.load_state_dict(model_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    dan.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return dan


def get_model(pretrained=False, **kwargs):
    """DAN 16-layer model (configuration "VGG16")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    dan = DAN(make_layers(backbone['VGG16'], batch_norm=True), **kwargs)

    if pretrained:
        pretrained_dict = torch.load("../pre_trained_weights/vgg16_bn-6c64b313.pth")
        model_dict = dan.state_dict()
        # 1. filter out unnecessary keys -------------------------------------------------------------------------------
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict --------------------------------------------------------------
        model_dict.update(pretrained_dict)
        dan.load_state_dict(model_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    dan.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return dan


if __name__ == "__main__":
    model = get_aud_linear_regressor()
    x = torch.randn(2, 79534).cuda()
    y = model(x)
    print(y.shape)

    # model = get_dan_model(pretrained=True)
    # x = torch.randn(2, 3, 244, 244).cuda()
    # y = model(x)
    # print(y, y.shape)

"""
questions:
    1) concatenate or add ,if add more weights saved
    2) hidden layers 50176 --> 1024 --> 5, mapping efficient
    3) L2 norm vs batch norm
    4) dropout or not
    5) freeze batch norm or not when training 
    6) pre-trained models from imagenet or face-net 
"""
