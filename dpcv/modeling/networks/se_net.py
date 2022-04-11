import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from dpcv.modeling.module.resnet_tv import ResNet
from dpcv.modeling.module.se_resnet import SEBottleneck
from .build import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
def se_resnet50(cfg, num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        num_classes (int): number of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Note:
        the resnet use sigmoid function for the out fc layer's output since the
        personality label in range (0, 1)
    """
    num_classes = cfg.MODEL.NUM_CLASS if cfg.MODEL.NUM_CLASS is not None else num_classes

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, return_feat=cfg.MODEL.RETURN_FEATURE)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if cfg.MODEL.PRETRAIN:
        # model.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))

        pretrained_dict = load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@NETWORK_REGISTRY.register()
def se_resnet50_true_personality(cfg, num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        num_classes (int): number of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Note:
        the resnet use sigmoid function for the out fc layer's output since the
        personality label in range (0, 1)
    """
    num_classes = cfg.MODEL.NUM_CLASS if cfg.MODEL.NUM_CLASS is not None else num_classes

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, sigmoid_output=False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if cfg.MODEL.PRETRAIN:
        # model.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))

        pretrained_dict = load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model