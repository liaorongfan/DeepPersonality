import torch
import torch.nn as nn
from dpcv.modeling.module.bi_modal_resnet_module import AudioVisualResNet, AudInitStage
from dpcv.modeling.module.bi_modal_resnet_module import BiModalBasicBlock, VisInitStage
from dpcv.modeling.module.bi_modal_resnet_module import aud_conv1x9, aud_conv1x1, vis_conv3x3, vis_conv1x1
from dpcv.modeling.module.weight_init_helper import initialize_weights
from .build import NETWORK_REGISTRY


class CRNet(nn.Module):
    def __init__(self, only_train_guider=False):
        super(CRNet, self).__init__()
        self.train_guider = only_train_guider
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

    def train_classifier(self):
        self.train_guider = True

    def train_regressor(self):
        self.train_guider = False

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
        cls_guide = torch.stack([glo_cls + loc_cls + wav_cls], dim=-1).mean(dim=-1).squeeze()
        if self.train_guider:
            return cls_guide
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

        return cls_guide, out


class CRNet2(nn.Module):
    def __init__(self, init_weights=True, return_feat=False):
        super(CRNet2, self).__init__()
        self.train_guider_epo = 1
        self.return_feature = return_feat
        self.train_regressor = False

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

        if init_weights:
            initialize_weights(self)

    def set_train_classifier_epo(self, epo):
        self.train_guider_epo = epo

    def set_train_regressor(self):
        self.train_regressor = True

    def forward(self, global_img, local_img, audio_wav):
        glo_feature = self.global_img_branch(global_img)  # (bs, 512, 2, 2)
        loc_feature = self.local_img_branch(local_img)    # (bs, 512, 2, 2)
        aud_feature = self.audio_branch(audio_wav)        # (bs, 512, 1, 4)
        # ---- first training stage class guide -----
        glo_cls = self.global_cls_guide(glo_feature)      # (bs, 5, 4)
        loc_cls = self.local_cls_guide(loc_feature)       # (bs, 5, 4)
        wav_cls = self.wav_cls_guide(aud_feature)         # (bs, 5, 4)

        glo_cls = glo_cls.view(glo_cls.size(0), 5, -1)
        loc_cls = loc_cls.view(loc_cls.size(0), 5, -1)
        wav_cls = wav_cls.view(wav_cls.size(0), 5, -1)
        cls_guide = torch.stack([glo_cls + loc_cls + wav_cls], dim=-1).mean(dim=-1).squeeze()
        if not self.train_regressor:
            return cls_guide
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
        if self.return_feature:
            return cls_guide, out, out_reg
        return cls_guide, out


class CRNetAud(nn.Module):
    def __init__(self):
        super(CRNetAud, self).__init__()
        self.train_guider_epo = 30  # default train 50 epochs for classification guidence
        self.train_regressor = False
        self.audio_branch = AudioVisualResNet(
            in_channels=1, init_stage=AudInitStage,
            block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
            layers=[3, 4, 6, 3],
            out_spatial=(1, 4)
        )

        self.wav_cls_guide = nn.Conv2d(512, 20, (1, 4))
        self.out_map = nn.Linear(512, 1)

    def set_train_classifier_epo(self, epo):
        self.train_guider_epo = epo

    def set_train_regressor(self):
        self.train_regressor = True

    def forward(self, audio_wav):
        aud_feature = self.audio_branch(audio_wav)
        # ---- first training stage class guide -----
        wav_cls = self.wav_cls_guide(aud_feature)

        wav_cls = wav_cls.view(wav_cls.size(0), 5, -1)
        cls_guide = wav_cls  # torch.stack([wav_cls], dim=-1).mean(dim=-1).squeeze()
        if not self.train_regressor:
            return wav_cls
        # --- second training stage guided regress ---
        wav_cls_feature = aud_feature.view(aud_feature.size(0), 512, 4).permute(0, 2, 1)
        wav_cls_score = torch.softmax(wav_cls, -1)
        guided_wav_reg = torch.matmul(wav_cls_score, wav_cls_feature)  # where i = {1,2,3,4,5}

        out_reg = guided_wav_reg
        out = self.out_map(out_reg)
        out = out.view(out.size(0), -1)

        return cls_guide, out


class CRNetVis(nn.Module):
    def __init__(self, init_weights=True, return_feat=False):
        super(CRNetVis, self).__init__()
        self.train_guider_epo = 1
        self.return_feature = return_feat
        self.train_regressor = False

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
        # self.audio_branch = AudioVisualResNet(
        #     in_channels=1, init_stage=AudInitStage,
        #     block=BiModalBasicBlock, conv=[aud_conv1x9, aud_conv1x1],
        #     layers=[3, 4, 6, 3],
        #     out_spatial=(1, 4)
        # )

        self.global_cls_guide = nn.Conv2d(512, 20, 2)
        self.local_cls_guide = nn.Conv2d(512, 20, 2)
        # self.wav_cls_guide = nn.Conv2d(512, 20, (1, 4))
        self.out_map = nn.Linear(512, 1)

        if init_weights:
            initialize_weights(self)

    def set_train_classifier_epo(self, epo):
        self.train_guider_epo = epo

    def set_train_regressor(self):
        self.train_regressor = True

    def forward(self, global_img, local_img):
        glo_feature = self.global_img_branch(global_img)  # (bs, 512, 2, 2)
        loc_feature = self.local_img_branch(local_img)    # (bs, 512, 2, 2)
        # aud_feature = self.audio_branch(audio_wav)        # (bs, 512, 1, 4)
        # ---- first training stage class guide -----
        glo_cls = self.global_cls_guide(glo_feature)      # (bs, 5, 4)
        loc_cls = self.local_cls_guide(loc_feature)       # (bs, 5, 4)
        # wav_cls = self.wav_cls_guide(aud_feature)         # (bs, 5, 4)

        glo_cls = glo_cls.view(glo_cls.size(0), 5, -1)
        loc_cls = loc_cls.view(loc_cls.size(0), 5, -1)
        # wav_cls = wav_cls.view(wav_cls.size(0), 5, -1)
        cls_guide = torch.stack([glo_cls + loc_cls], dim=-1).mean(dim=-1).squeeze()
        if not self.train_regressor:
            return cls_guide
        # --- second training stage guided regress ---
        glo_cls_feature = glo_feature.view(glo_feature.size(0), 512, 4).permute(0, 2, 1)
        loc_cls_feature = loc_feature.view(loc_feature.size(0), 512, 4).permute(0, 2, 1)
        # wav_cls_feature = aud_feature.view(aud_feature.size(0), 512, 4).permute(0, 2, 1)

        glo_cls_score = torch.softmax(glo_cls, -1)
        loc_cls_score = torch.softmax(loc_cls, -1)
        # wav_cls_score = torch.softmax(wav_cls, -1)

        guided_glo_reg = torch.matmul(glo_cls_score, glo_cls_feature)  # (_, 5, 4) matmul (_, 4, 512) = (_, 5, 512)
        guided_loc_reg = torch.matmul(loc_cls_score, loc_cls_feature)  # every dim in axis 1 is a weighted sum of P_i
        # guided_wav_reg = torch.matmul(wav_cls_score, wav_cls_feature)  # where i = {1,2,3,4,5}

        out_reg = guided_glo_reg + guided_loc_reg # + guided_wav_reg
        out = self.out_map(out_reg)
        out = out.view(out.size(0), -1)
        if self.return_feature:
            return cls_guide, out, out_reg
        return cls_guide, out


def get_crnet_model(only_train_guider=True):
    cr_net = CRNet(only_train_guider)
    cr_net.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return cr_net


@NETWORK_REGISTRY.register()
def crnet_model(cfg=None):
    cr_net = CRNet2(return_feat=cfg.MODEL.RETURN_FEATURE)
    return cr_net.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@NETWORK_REGISTRY.register()
def get_crnet_aud_model(cfg):
    cr_net_aud = CRNetAud()
    return cr_net_aud.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@NETWORK_REGISTRY.register()
def get_crnet_vis_model(cfg):
    cr_net_vis = CRNetVis()
    return cr_net_vis.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    import torch

    global_img_input = torch.randn(2, 3, 224, 224)
    local_img_input = torch.randn(2, 3, 112, 112)
    wav_input = torch.randn(2, 1, 1, 244832)
    # model = CRNet(only_train_guider=True)
    # y = model(global_img_input, local_img_input, wav_input)
    # model = CRNet2()
    model = CRNetVis()
    # y = model(global_img_input, local_img_input, wav_input)
    y = model(global_img_input, local_img_input, wav_input)
    print(y.shape)

    model.set_train_regressor()
    # cls, reg = model(global_img_input, local_img_input, wav_input)
    cls, reg = model(global_img_input, local_img_input)
    # cls, reg = model(wav_input)
    print(cls.shape, reg.shape)
