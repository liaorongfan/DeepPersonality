import torch
import torch.nn as nn
from dpcv.modeling.module.bi_modal_resnet_module import AudioVisualResNet, AudInitStage
from dpcv.modeling.module.bi_modal_resnet_module import VisInitStage, BiModalBasicBlock
from dpcv.modeling.module.bi_modal_resnet_module import aud_conv1x9, aud_conv1x1, vis_conv3x3, vis_conv1x1


class AudioVisualResNet18(nn.Module):

    def __init__(self):
        super(AudioVisualResNet18, self).__init__()
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

    def forward(self, aud_input, vis_input):
        aud_x = self.audio_branch(aud_input)
        vis_x = self.visual_branch(vis_input)

        aud_x = aud_x.view(aud_x.size(0), -1)
        vis_x = vis_x.view(vis_x.size(0), -1)

        x = torch.cat([aud_x, vis_x], dim=-1)
        x = self.linear(x)
        x = torch.tanh(x)
        return x


def get_audiovisual_resnet_model():
    multi_modal_model = AudioVisualResNet18()
    multi_modal_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return multi_modal_model


if __name__ == "__main__":
    aud = torch.randn(2, 1, 1, 50176)
    vis = torch.randn(2, 3, 224, 224)
    multi_model = AudioVisualResNet18()
    y = multi_model(aud, vis)
    print(y)

