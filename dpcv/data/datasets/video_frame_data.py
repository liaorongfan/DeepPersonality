import random
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import glob
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY


class SingleFrameData(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans

    def __getitem__(self, index):
        img = self.get_image_data(index)
        label = self.get_ocean_label(index)

        if self.trans:
            img = self.trans(img)

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


class AllSampleFrameData(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None, length=100):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans
        self.len = length

    def __getitem__(self, idx):
        img_obj_ls = self.get_sample_frames(idx)
        label = self.get_ocean_label(idx)
        if self.trans is not None:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

    def get_sample_frames(self, idx):
        img_dir = self.img_dir_ls[idx]
        # Note randomly ordered after glob search
        img_path_ls = glob.glob(f"{img_dir}/*.jpg")
        # downsample the frames to 100 / video
        sample_frames_id = np.linspace(0, len(img_path_ls), self.len, endpoint=False, dtype=np.int16).tolist()
        img_path_ls_sampled = [img_path_ls[idx] for idx in sample_frames_id]
        img_obj_ls = [Image.open(img_path) for img_path in img_path_ls_sampled]
        return img_obj_ls


class AllSampleFrameData2(VideoData):
    def __init__(self, data_root, img_dir, label_file, trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.trans = trans

    def __getitem__(self, idx):
        img_obj_ls = self.get_sample_frames(idx)
        label = self.get_ocean_label(idx)
        if self.trans is not None:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        return {"all_images": img_obj_ls, "label": torch.as_tensor(label)}

    def get_sample_frames(self, idx):
        img_dir = self.img_dir_ls[idx]
        # Note randomly ordered after glob search
        img_path_ls = sorted(glob.glob(f"{img_dir}/*.jpg"))
        img_obj_ls = [Image.open(img_path) for img_path in img_path_ls]
        return img_obj_ls


def make_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]),\
        "'mode' should be 'train' , 'valid' 'trainval' 'test', 'full_test' "
    transform = set_transform_op()
    if mode == "train":
        data_set = SingleFrameData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = SingleFrameData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            transform,
        )
    elif mode == "trainval":
        data_set = SingleFrameData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            transform,
        )
    else:
        data_set = SingleFrameData(
            cfg.DATA_ROOT,
            cfg.TEST_IMG_DATA,
            cfg.TEST_LABEL_DATA,
            transform,
        )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def single_frame_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE
    transform = build_transform_spatial(cfg)
    if mode == "train":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transform,
        )
        shuffle = False
    elif mode == "trainval":
        data_set = SingleFrameData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAINVAL_IMG_DATA,
            cfg.DATA.TRAINVAL_LABEL_DATA,
            transform,
        )
    elif mode == "full_test":
        return AllSampleFrameData(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
    else:
        data_set = SingleFrameData(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
        shuffle = False

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
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
