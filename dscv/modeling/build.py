import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from ..modeling.backbone.vgg_tv import vgg16_bn
from ..modeling.backbone.se_resnet import se_resnet50


def get_model(cfg, cls_num, logger):
    """
    创建模型
    :param cfg:
    :param cls_num:
    :return:
    """
    if cfg.model_name == "resnet18":
        model = resnet18()
        if os.path.exists(cfg.path_resnet18):
            pretrained_state_dict = torch.load(cfg.path_resnet18, map_location="cpu")
            model.load_state_dict(pretrained_state_dict)    # load pretrain model
            logger.info("load pretrained model!")
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cls_num)  # 102
    elif cfg.model_name == "vgg16_bn":
        model = vgg16_bn()
        if os.path.exists(cfg.path_vgg16bn):
            pretrained_state_dict = torch.load(cfg.path_vgg16bn, map_location="cpu")
            model.load_state_dict(pretrained_state_dict)    # load pretrain model
            logger.info("load pretrained model!")
        # 替换网络层
        in_feat_num = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat_num, cls_num)
    elif cfg.model_name == "se_resnet50":
        model = se_resnet50()
        if os.path.exists(cfg.path_se_res50):
            model.load_state_dict(torch.load(cfg.path_se_res50))    # load pretrain model
            logger.info("load pretrained model!")
        in_feat_num = model.fc.in_features
        model.fc = nn.Linear(in_feat_num, cls_num)
    else:
        raise Exception("Invalid model name. got {}".format(cfg.model_name))
    return model
