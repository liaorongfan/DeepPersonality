import torch
from torch.utils.data import DataLoader
import glob
import numpy as np
import os
from pathlib import Path
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop,  TemporalDownsample, TemporalEvenCropDownsample
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.datasets.common import VideoLoader
from dpcv.data.datasets.ture_personality_data import Chalearn21FrameData


class VideoFrameSegmentData(VideoData):
    """ Dataloader for 3d models, (3d_resnet, slow-fast, tpn, vat)

    """
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
        imgs = self.frame_sample(img_dir)
        return imgs

    def frame_sample(self, img_dir):
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


class FullTestVideoSegmentData(VideoFrameSegmentData):

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
            image_segment_obj = torch.stack(image_segment_obj, 0).permute(1, 0, 2, 3)
            image_segment_obj_ls.append(image_segment_obj)
        return image_segment_obj_ls


class TruePersonalityVideoFrameSegmentData(Chalearn21FrameData):
    """ Dataloader for 3d models, (3d_resnet, slow-fast, tpn, vat)

    """
    def __init__(self, data_root, data_split, task, data_type, video_loader, spa_trans=None, tem_trans=None):
        super().__init__(data_root, data_split, task, data_type, even_downsample=2000, trans=None, segment=True)
        self.loader = video_loader
        self.spa_trans = spa_trans
        self.tem_trans = tem_trans

    def __getitem__(self, index):
        img = self.get_image_data(index)
        label = self.get_image_label(index)
        return {"image": img, "label": torch.as_tensor(label, dtype=torch.float32)}

    def __len__(self):
        return len(self.img_dir_ls)

    def get_image_data(self, index):
        img_dir = self.img_dir_ls[index]
        imgs = self.frame_sample(img_dir)
        return imgs

    def get_image_label(self, index):
        img_dir = self.img_dir_ls[index]
        session, part = img_dir.split("/")
        if self.type == "face":
            part = part.replace("_face", "")
        part = part.replace(self.task_mark, "T")
        participant_id = self.session_id[str(int(session))][part]
        participant_trait = self.parts_personality[participant_id]
        participant_trait = np.array([float(v) for v in participant_trait.values()])
        return participant_trait

    def frame_sample(self, img_dir):
        img_dir = os.path.join(self.data_dir, img_dir)
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

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test' or 'full_test' "

    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalDownsample(length=100), TemporalRandomCrop(16)]
    # temporal_transform = [TemporalRandomCrop(16)]
    # temporal_transform = [TemporalDownsample(32)]

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
    elif mode == "full_test":
        temporal_transform = [TemporalDownsample(length=100), TemporalEvenCropDownsample(16, 6)]
        temporal_transform = TemporalCompose(temporal_transform)
        return FullTestVideoSegmentData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
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


@DATA_LOADER_REGISTRY.register()
def true_personality_spatial_temporal_data_loader(cfg, mode="train"):
    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalRandomCrop(32)]
    # temporal_transform = [TemporalDownsample(length=2000), TemporalRandomCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA

    if data_cfg.TYPE == "face":
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    data_set = TruePersonalityVideoFrameSegmentData(
        data_root="datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,
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
    from dpcv.config.default_config_opt import cfg

    os.chdir("../../../")
    print(os.getcwd())

    # interpret_data = InterpretData(
    #     data_root="datasets",
    #     img_dir="image_data/valid_data",
    #     label_file="annotation/annotation_validation.pkl",
    #     trans=set_transform_op(),
    # )
    # print(interpret_data[18])

    # data_loader = make_data_loader(cfg, mode="valid")
    # for i, item in enumerate(data_loader):
    #     print(item["image"].shape, item["label"].shape)
    #
    #     if i > 5:
    #         break

    train_dataset = true_personality_spatial_temporal_data_loader(cfg)
    print(len(train_dataset))
