import random

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import glob
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.datasets.transforms import set_transform_op


class InterpretData(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans

    def __getitem__(self, index):
        img = self.get_image_data(index)
        label = self.get_ocean_label(index)

        if self.trans:
            img = self.trans(img)

        # img = torch.as_tensor(img)
        # label = torch.as_tensor(label)

        return {"image": img, "label": torch.as_tensor(label)}

    def get_image_data(self, index):
        img_dir = self.img_dir_ls[index]
        img_path = self.image_sample(img_dir)
        img = Image.open(img_path).convert("RGB")
        return img

    @staticmethod
    def image_sample(img_dir):
        img_path_ls = glob.glob(f"{img_dir}/*.jpg")
        num_img = len(img_path_ls)
        # downsample the frames to 100 / video
        sample_frames = np.linspace(0, num_img, 100, endpoint=False, dtype=np.int16)
        selected = random.choice(sample_frames)
        return img_path_ls[selected]


def make_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    transform = set_transform_op()
    if mode == "train":
        data_set = InterpretData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = InterpretData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            transform,
        )
    elif mode == "trainval":
        data_set = InterpretData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            transform,
        )
    else:
        data_set = InterpretData(
            cfg.DATA_ROOT,
            cfg.TEST_IMG_DATA,
            cfg.TEST_LABEL_DATA,
            transform,
        )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )
    return data_loader


if __name__ == "__main__":
    import os
    from dpcv.config.interpret_dan_cfg import cfg
    os.chdir("../../")

    # interpret_data = InterpretData(
    #     data_root="datasets",
    #     img_dir="image_data/valid_data",
    #     label_file="annotation/annotation_validation.pkl",
    #     trans=set_transform_op(),
    # )
    # print(interpret_data[18])

    data_loader = make_data_loader(cfg, mode="valid")
    for i, item in enumerate(data_loader):
        print(item["image"].shape, item["label"].shape)

        if i > 5:
            break
