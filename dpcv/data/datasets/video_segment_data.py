import torch
from torch.utils.data import DataLoader
import glob
import numpy as np
from pathlib import Path
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.transforms.build import build_transform_opt
from .build import DATA_LOADER_REGISTRY
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop, TemporalEvenCrop, TemporalDownsample
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.datasets.common import VideoLoader


class VideoFrameSegmentData(VideoData):
    def __init__(self, data_root, img_dir, label_file, video_loader, spa_trans=None, tem_trans=None):
        super().__init__(data_root, img_dir, label_file)
        self.loader = video_loader
        self.spa_trans = spa_trans
        self.tem_trans = tem_trans

    def __getitem__(self, index):
        img = self.get_image_data(index)
        label = self.get_ocean_label(index)
        return {"image": img, "label": torch.as_tensor(label)}

    def get_image_data(self, index):
        img_dir = self.img_dir_ls[index]
        imgs = self.image_sample(img_dir)
        return imgs

    def image_sample(self, img_dir):
        if "face" in img_dir:
            frame_indices = self.list_face_frames(img_dir)
        else:
            frame_indices = self.list_frames(img_dir)

        if self.tem_trans is not None:
            frame_indices = self.tem_trans(frame_indices)
        imgs = self._loading(img_dir, frame_indices)
        return imgs

    def _loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spa_trans is not None:
            # more flexible image preprocess but for simple comparison we just use crop
            # self.spa_trans.randomize_parameters()
            clip = [self.spa_trans(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    @staticmethod
    def list_frames(img_dir):
        img_path_ls = glob.glob(f"{img_dir}/*.jpg")
        img_path_ls = sorted(img_path_ls, key=lambda x: int(Path(x).stem[6:]))
        frame_indices = [int(Path(path).stem[6:]) for path in img_path_ls]
        return frame_indices

    @staticmethod
    def list_face_frames(img_dir):
        img_path_ls = glob.glob(f"{img_dir}/*.jpg")
        img_path_ls = sorted(img_path_ls, key=lambda x: int(Path(x).stem[5:]))
        frame_indices = [int(Path(path).stem[5:]) for path in img_path_ls]
        return frame_indices


def make_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = set_transform_op()
    temporal_transform = [TemporalRandomCrop(32)]
    temporal_transform = TemporalCompose(temporal_transform)
    video_loader = VideoLoader()

    if mode == "train":
        data_set = VideoFrameSegmentData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = VideoFrameSegmentData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = VideoFrameSegmentData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = VideoFrameSegmentData(
            cfg.DATA_ROOT,
            cfg.TEST_IMG_DATA,
            cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def spatial_temporal_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = build_transform_opt(cfg)
    temporal_transform = [TemporalDownsample(), TemporalRandomCrop(16)]
    # temporal_transform = [TemporalRandomCrop(16)]

    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA
    if "face" in data_cfg.TRAIN_IMG_DATA:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    if mode == "train":
        data_set = VideoFrameSegmentData(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = VideoFrameSegmentData(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = VideoFrameSegmentData(
            data_cfg.ROOT,
            data_cfg.TRAINVAL_IMG_DATA,
            data_cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = VideoFrameSegmentData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )

    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        shuffle=loader_cfg.SHUFFLE,
        num_workers=loader_cfg.NUM_WORKERS,
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
