import torch
from torch.utils.data import DataLoader
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.transforms.build import build_transform_opt
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
from dpcv.data.datasets.common import VideoLoader

class SlowFastData(VideoFrameSegmentData):

    def __getitem__(self, index):

        img = self.get_image_data(index)
        frame_list = self.pack_pathway_output(img)
        label = self.get_ocean_label(index)
        return {"image": frame_list, "label": torch.as_tensor(label)}

    @staticmethod
    def pack_pathway_output(frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]

        return frame_list


def make_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = set_transform_op()
    temporal_transform = [TemporalRandomCrop(64)]
    temporal_transform = TemporalCompose(temporal_transform)
    video_loader = VideoLoader()

    if mode == "train":
        data_set = SlowFastData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = SlowFastData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = SlowFastData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = SlowFastData(
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
def slow_fast_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = build_transform_opt(cfg)
    temporal_transform = [TemporalRandomCrop(64)]
    temporal_transform = TemporalCompose(temporal_transform)
    video_loader = VideoLoader()

    data_cfg = cfg.DATA
    if mode == "train":
        data_set = SlowFastData(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = SlowFastData(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = SlowFastData(
            data_cfg.ROOT,
            data_cfg.TRAINVAL_IMG_DATA,
            data_cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = SlowFastData(
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
        print(item["image"][0].shape, item["image"][1].shape, item["label"].shape)