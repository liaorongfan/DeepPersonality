import torch
import torch.nn as nn
from dpcv.modeling.networks.dan import make_layers, backbone
# import torch.utils.model_zoo as model_zoo


class InterpretDAN(nn.Module):

    def __init__(self, features, num_classes=5, init_weights=True):
        super(InterpretDAN, self).__init__()
        self.features = features
        self.glo_ave_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.glo_ave_pooling(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = torch.sigmoid(x)  # since the regression range always fall in (0, 1)
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


if __name__ == "__main__":
    import os
    os.chdir("../../")
    model = get_interpret_dan_model(pretrained=True)
    x = torch.randn(2, 3, 244, 244).cuda()
    y = model(x)
    print(y, y.shape)

