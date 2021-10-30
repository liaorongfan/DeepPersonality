import torch
from torch.utils.data import DataLoader
from data.transforms.transform import set_tpn_transform_op
from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData


class TPNData(VideoFrameSegmentData):

    def __getitem__(self, index):

        img = self.get_image_data(index)
        label = self.get_ocean_label(index)
        return {"image": img, "label": torch.as_tensor(label)}

    def _loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spa_trans is not None:
            clip = [self.spa_trans(img) for img in clip]
        clip = torch.stack(clip, 0)
        return clip


def make_data_loader(cfg, mode="train"):
    from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop
    from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
    from dpcv.data.datasets.common import VideoLoader

    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = set_tpn_transform_op()
    temporal_transform = [TemporalRandomCrop(8)]
    temporal_transform = TemporalCompose(temporal_transform)
    video_loader = VideoLoader()

    if mode == "train":
        data_set = TPNData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = TPNData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = TPNData(
            cfg.DATA_ROOT,
            cfg.TRAINVAL_IMG_DATA,
            cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = TPNData(
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


if __name__ == "__main__":
    import os
    from dpcv.config.tpn_cfg import cfg

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
