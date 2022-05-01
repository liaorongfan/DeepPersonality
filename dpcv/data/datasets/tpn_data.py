import torch
from torch.utils.data import DataLoader
from dpcv.data.transforms.transform import set_tpn_transform_op
from dpcv.data.datasets.video_segment_data import VideoFrameSegmentData
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop, TemporalDownsample, TemporalEvenCropDownsample
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.datasets.common import VideoLoader
from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
from dpcv.data.datasets.video_segment_data import TruePersonalityVideoFrameSegmentData



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


class FullTestTPNData(TPNData):

    def __getitem__(self, index):

        imgs = self.get_image_data(index)
        label = self.get_ocean_label(index)
        return {"all_images": imgs, "label": torch.as_tensor(label)}

    def frame_sample(self, img_dir):

        if "face" in img_dir:
            frame_indices = self.list_face_frames(img_dir)
        else:
            frame_indices = self.list_frames(img_dir)

        if self.tem_trans is not None:
            frame_indices = self.tem_trans(frame_indices)
        imgs = self.load_batch_images(img_dir, frame_indices)
        return imgs

    def load_batch_images(self, img_dir, frame_indices_ls):
        image_segment_obj_ls = []
        for frame_seg_idx in frame_indices_ls:
            image_segment_obj = self.loader(img_dir, frame_seg_idx)
            if self.spa_trans is not None:
                image_segment_obj = [self.spa_trans(img) for img in image_segment_obj]
            image_segment_obj = torch.stack(image_segment_obj, 0)
            image_segment_obj_ls.append(image_segment_obj)
        return image_segment_obj_ls


class TPNTruePerData(TruePersonalityVideoFrameSegmentData):

    def _loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spa_trans is not None:
            clip = [self.spa_trans(img) for img in clip]
        clip = torch.stack(clip, 0)
        return clip


def make_data_loader(cfg, mode="train"):
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


@DATA_LOADER_REGISTRY.register()
def tpn_data_loader(cfg, mode="train"):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid' or 'trainval'"

    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalDownsample(length=100), TemporalRandomCrop(16)]
    # temporal_transform = [TemporalDownsample(length=16)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA
    if "face" in data_cfg.TRAIN_IMG_DATA:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    if mode == "train":
        data_set = TPNData(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "valid":
        data_set = TPNData(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "trainval":
        data_set = TPNData(
            data_cfg.ROOT,
            data_cfg.TRAINVAL_IMG_DATA,
            data_cfg.TRAINVAL_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    elif mode == "full_test":
        temporal_transform = [TemporalDownsample(length=100), TemporalEvenCropDownsample(16, 6)]
        temporal_transform = TemporalCompose(temporal_transform)
        return FullTestTPNData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
            video_loader,
            spatial_transform,
            temporal_transform,
        )
    else:
        data_set = TPNData(
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


@DATA_LOADER_REGISTRY.register()
def tpn_true_per_data_loader(cfg, mode="train"):
    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalRandomCrop(16)]
    # temporal_transform = [TemporalDownsample(length=2000), TemporalRandomCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA
    if data_cfg.TYPE == "face":
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    data_set = TPNTruePerData(
        data_root="datasets/chalearn2021",
        data_split=mode,
        task=data_cfg.SESSION,
        data_type=data_cfg.TYPE,
        video_loader=video_loader,
        spa_trans=spatial_transform,
        tem_trans=temporal_transform,
    )

    shuffle = True if mode == "train" else False
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=loader_cfg.NUM_WORKERS,
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
