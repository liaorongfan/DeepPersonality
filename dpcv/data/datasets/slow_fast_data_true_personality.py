from dpcv.data.datasets.video_segment_data import TruePersonalityVideoFrameSegmentData
import torch
from torch.utils.data import DataLoader
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
from dpcv.data.transforms.temporal_transforms import TemporalRandomCrop,  TemporalDownsample, TemporalEvenCropDownsample
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.datasets.common import VideoLoader


class SlowFastTruePerData(TruePersonalityVideoFrameSegmentData):

    def __getitem__(self, index):
        img = self.get_image_data(index)
        frame_list = self.pack_pathway_output(img)
        label = self.get_image_label(index)
        return {"image": frame_list, "label": torch.as_tensor(label, dtype=torch.float32)}

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


@DATA_LOADER_REGISTRY.register()
def true_per_slow_fast_data_loader(cfg, mode="train"):
    spatial_transform = build_transform_spatial(cfg)
    temporal_transform = [TemporalRandomCrop(64)]
    # temporal_transform = [TemporalDownsample(length=2000), TemporalRandomCrop(64)]
    temporal_transform = TemporalCompose(temporal_transform)

    data_cfg = cfg.DATA

    if data_cfg.TYPE == "face":
        video_loader = VideoLoader(image_name_formatter=lambda x: f"face_{x}.jpg")
    else:
        video_loader = VideoLoader(image_name_formatter=lambda x: f"frame_{x}.jpg")

    data_set = SlowFastTruePerData(
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
    from dpcv.config.default_config_opt import cfg
    os.chdir("../../../")
    dataloader = true_per_slow_fast_data_loader(cfg)
