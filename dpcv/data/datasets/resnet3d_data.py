import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import glob
from pathlib import Path
from dpcv.data.datasets.bi_modal_data import VideoData
from data.transforms.transform import set_transform_op


class VideoLoader:

    def __init__(self, image_name_formatter=lambda x: f"frame_{x}.jpg"):
        self.image_name_formatter = image_name_formatter

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_path, self.image_name_formatter(i))
            if os.path.exists(image_path):
                video.append(Image.open(image_path).convert("RGB"))
        return video


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



def make_data_loader(cfg, mode="train"):
    from dpcv.data.transforms.temporal_transforms import SlidingWindow, TemporalRandomCrop
    from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose

    assert (mode in ["train", "valid", "trainval", "test"]), "'mode' should be 'train' , 'valid' or 'trainval'"
    spatial_transform = set_transform_op()
    temporal_transform = [TemporalRandomCrop(16)]
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

        # if i > 5:
        #     break
