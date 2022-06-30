import torch
import torch.nn as nn
from dpcv.modeling.networks.dan import make_layers, backbone
from dpcv.modeling.networks.build import NETWORK_REGISTRY
from dpcv.modeling.module.weight_init_helper import initialize_weights
# import torch.utils.model_zoo as model_zoo


class InterpretDAN(nn.Module):

    def __init__(self, features, num_classes=5, init_weights=True, return_feat=False, use_sigmoid=True):
        super(InterpretDAN, self).__init__()
        self.features = features
        self.glo_ave_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if init_weights:
            initialize_weights(self)
        self.return_feature = return_feat
        self.use_sigmoid = use_sigmoid

    def forward(self, x):  # x (2, 3, 244, 244)
        x = self.features(x)         # x (2, 512, 7, 7)
        x = self.glo_ave_pooling(x)  # x (2, 512, 1, 1)
        feat = x.flatten(1)  # feat (2, 512)
        x = self.fc(feat)
        if self.use_sigmoid:
            x = torch.sigmoid(x)  # since the regression range always fall in (0, 1)
        if self.return_feature:
            return x, feat
        return x


def get_interpret_dan_model(cfg, pretrained=False, **kwargs):
    """Interpret DAN 16-layer model (configuration "VGG16")

    Args:
        cfg: config for interpret dan model
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    interpret_dan = InterpretDAN(make_layers(backbone['VGG16'], batch_norm=True), **kwargs)

    if pretrained:
        pretrained_dict = torch.load(cfg.PRE_TRAINED_MODEL)
        model_dict = interpret_dan.state_dict()
        # 1. filter out unnecessary keys -------------------------------------------------------------------------------
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict --------------------------------------------------------------
        model_dict.update(pretrained_dict)
        interpret_dan.load_state_dict(model_dict)
        # load pre_trained model from model_zoo for standard models in pytorch -----------------------------------------
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    interpret_dan.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return interpret_dan


@NETWORK_REGISTRY.register()
def interpret_dan_model(cfg):
    interpret_dan = InterpretDAN(
        make_layers(backbone['VGG16'], batch_norm=True), return_feat=cfg.MODEL.RETURN_FEATURE)
    interpret_dan.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return interpret_dan


@NETWORK_REGISTRY.register()
def interpret_dan_model_true_personality(cfg):
    interpret_dan = InterpretDAN(
        make_layers(backbone['VGG16'], batch_norm=True),
        return_feat=cfg.MODEL.RETURN_FEATURE,
        use_sigmoid=False
    )
    interpret_dan.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return interpret_dan


if __name__ == "__main__":
    import os
    os.chdir("../../")
    model = interpret_dan = InterpretDAN(make_layers(backbone['VGG16'], batch_norm=True)).cuda()
    x = torch.randn(2, 3, 244, 244).cuda()
    y = model(x)
    print(y, y.shape)

