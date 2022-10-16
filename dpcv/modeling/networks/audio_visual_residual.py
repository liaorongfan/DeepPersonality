import torch
import torch.nn as nn
from dpcv.modeling.module.bi_modal_resnet_module import AudioVisualResNet, AudInitStage
from dpcv.modeling.module.bi_modal_resnet_module import VisInitStage, BiModalBasicBlock
from dpcv.modeling.module.bi_modal_resnet_module import aud_conv1x9, aud_conv1x1, vis_conv3x3, vis_conv1x1
from dpcv.modeling.module.weight_init_helper import initialize_weights
from dpcv.modeling.networks.build import NETWORK_REGISTRY


class AudioVisualResNet18(nn.Module):

    def __init__(self, init_weights=True, return_feat=False):
        super(AudioVisualResNet18, self).__init__()
        self.return_feature = return_feat
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage,
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.visual_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(512, 5)

        if init_weights:
            initialize_weights(self)

    def forward(self, aud_input, vis_input):
        aud_x = self.audio_branch(aud_input)
        vis_x = self.visual_branch(vis_input)

        aud_x = aud_x.view(aud_x.size(0), -1)
        vis_x = vis_x.view(vis_x.size(0), -1)

        feat = torch.cat([aud_x, vis_x], dim=-1)
        x = self.linear(feat)
        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature:
            return x, feat
        return x


class VisualResNet18(nn.Module):

    def __init__(self, init_weights=True, return_feat=False):
        super(VisualResNet18, self).__init__()
        self.return_feature = return_feat

        self.visual_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(256, 5)

        if init_weights:
            initialize_weights(self)

    def forward(self, vis_input):
        # aud_x = self.audio_branch(aud_input)
        vis_x = self.visual_branch(vis_input)

        # aud_x = aud_x.view(aud_x.size(0), -1)
        vis_x = vis_x.view(vis_x.size(0), -1)

        feat = vis_x
        x = self.linear(vis_x)
        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = (x + 1) / 2  # scale tanh output to [0, 1]
        if self.return_feature:
            return x, feat
        return x


class AudioResNet18(nn.Module):

    def __init__(self):
        super(AudioResNet18, self).__init__()
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage,
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
            channels=[32, 64, 128, 256],
            layers=[2, 2, 2, 2]
        )
        self.linear = nn.Linear(256, 5)

    def forward(self, aud_input):
        aud_x = self.audio_branch(aud_input)
        aud_x = aud_x.view(aud_x.size(0), -1)
        x = self.linear(aud_x)
        x = torch.sigmoid(x)
        return x


@NETWORK_REGISTRY.register()
def audiovisual_resnet(cfg=None):
    multi_modal_model = AudioVisualResNet18(return_feat=cfg.MODEL.RETURN_FEATURE)
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


def get_audiovisual_resnet_model():
    multi_modal_model = AudioVisualResNet18()
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


@NETWORK_REGISTRY.register()
def get_audio_resnet_model(cfg=None):
    aud_modal_model = AudioResNet18()
    aud_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return aud_modal_model


@NETWORK_REGISTRY.register()
def get_visual_resnet_model(cfg=None):
    visual_modal_model = VisualResNet18()
    visual_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return visual_modal_model


if __name__ == "__main__":
    aud = torch.randn(2, 1, 1, 50176)
    vis = torch.randn(2, 3, 224, 224)
    # multi_model = AudioVisualResNet18()
    # y = multi_model(aud, vis)
    model = AudioResNet18()
    y = model(aud)
    print(y)

