import torch.nn as nn
from dpcv.modeling.networks.bi_modal_resnet_module import AudioVisualResNet, AudInitStage
from dpcv.modeling.networks.bi_modal_resnet_module import BiModalBasicBlock, VisInitStage
from dpcv.modeling.networks.bi_modal_resnet_module import aud_conv1x9, aud_conv1x1, vis_conv3x3, vis_conv1x1


# import torch.utils.model_zoo as model_zoo
# model_urls = {
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
# }


class CRNet(nn.Module):
    def __init__(self, train_guider=False):
        super(CRNet, self).__init__()
        self.train_guider = train_guider
        self.global_img_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            layers=[3, 4, 6, 3],  # layer setting of resnet34
            out_spatial=(2, 2)
        )
        self.local_img_branch = AudioVisualResNet(
            in_channels=3, init_stage=VisInitStage,
            block=BiModalBasicBlock, conv=[vis_conv3x3, vis_conv1x1],
            layers=[3, 4, 6, 3],
            out_spatial=(2, 2)
        )
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage,
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
            layers=[3, 4, 6, 3],
            out_spatial=(1, 4)
        )

        self.global_cls_guide = nn.Conv2d(512, 20, 2)
        self.local_cls_guide = nn.Conv2d(512, 20, 2)
        self.wav_cls_guide = nn.Conv2d(512, 20, (1, 4))
        self.out_map = nn.Linear(512, 1)

    def forward(self, global_img, local_img, audio_wav):
        glo_feature = self.global_img_branch(global_img)
        loc_feature = self.local_img_branch(local_img)
        aud_feature = self.audio_branch(audio_wav)
        # ---- first training stage class guide -----
        glo_cls = self.global_cls_guide(glo_feature)
        loc_cls = self.local_cls_guide(loc_feature)
        wav_cls = self.wav_cls_guide(aud_feature)

        glo_cls = glo_cls.view(glo_cls.size(0), 5, -1)
        loc_cls = loc_cls.view(loc_cls.size(0), 5, -1)
        wav_cls = wav_cls.view(wav_cls.size(0), 5, -1)
        if self.train_guider:
            return glo_cls, loc_cls, wav_cls
        # --- second training stage guided regress ---
        glo_cls_feature = glo_feature.view(glo_feature.size(0), 512, 4).permute(0, 2, 1)
        loc_cls_feature = loc_feature.view(loc_feature.size(0), 512, 4).permute(0, 2, 1)
        wav_cls_feature = aud_feature.view(aud_feature.size(0), 512, 4).permute(0, 2, 1)

        glo_cls_score = torch.softmax(glo_cls, -1)
        loc_cls_score = torch.softmax(loc_cls, -1)
        wav_cls_score = torch.softmax(wav_cls, -1)

        guided_glo_reg = torch.matmul(glo_cls_score, glo_cls_feature)  # (_, 5, 4) matmul (_, 4, 512) = (_, 5, 512)
        guided_loc_reg = torch.matmul(loc_cls_score, loc_cls_feature)  # every dim in axis 1 is a weighted sum of P_i
        guided_wav_reg = torch.matmul(wav_cls_score, wav_cls_feature)  # where i = {1,2,3,4,5}

        out_reg = guided_glo_reg + guided_loc_reg + guided_wav_reg
        out = self.out_map(out_reg)
        out = out.view(out.size(0), -1)

        return glo_cls, loc_cls, wav_cls, out
        # return glo_feature.shape, loc_feature.shape, aud_feature.shape, \
        #        glo_cls.shape, loc_cls.shape, wav_cls.shape, \
        #        glo_cls_score.shape, loc_cls_score.shape, wav_cls_score.shape, \
        #         glo_cls_feature.shape, loc_cls_feature.shape, wav_cls_feature.shape, \
        #         guided_glo_reg.shape, guided_loc_reg.shape, guided_wav_reg.shape, \
        #         out.shape


if __name__ == "__main__":
    import torch

    global_img_input = torch.randn(2, 3, 112, 112)
    local_img_input = torch.randn(2, 3, 112, 112)
    wav_input = torch.randn(2, 1, 1, 244832)
    model = CRNet(train_guider=True)
    y = model(global_img_input, local_img_input, wav_input)
    for item in y:
        print(item.shape)
    model.train_guider = False

    y = model(global_img_input, local_img_input, wav_input)
    for item in y:
        print(item.shape)
