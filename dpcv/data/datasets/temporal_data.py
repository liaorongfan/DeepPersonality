"""
TODO: merge temporal data to bi_modal_data
"""
import torch
import os
import glob
from dpcv.data.datasets.bi_modal_data import VideoData
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import random
import numpy as np
from dpcv.data.transforms.transform import face_image_transform
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY


class TemporalData(VideoData):
    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None):
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform

    def __getitem__(self, idx):
        anno_score = self.get_ocean_label(idx)
        imgs_array_ls = self._get_statistic_img_sample(idx)
        wav_ft = self._get_wav_sample(idx)
        if self.transform:
            imgs_ten_ls = []
            for img_arr in imgs_array_ls:
                img_ten = self.transform(img_arr)
                imgs_ten_ls.append(img_ten)
            imgs_ten = torch.stack(imgs_ten_ls, dim=0)
        else:
            imgs_ten = torch.as_tensor(imgs_array_ls)

        wav_ft = torch.as_tensor(wav_ft, dtype=imgs_ten.dtype)
        anno_score = torch.as_tensor(anno_score, dtype=imgs_ten.dtype)
        sample = {"image": imgs_ten, "audio": wav_ft, "label": anno_score}
        return sample

    def __len__(self):
        return len(self.img_dir_ls)

    def _get_statistic_img_sample(self, index):
        imgs = glob.glob(self.img_dir_ls[index] + "/*.jpg")
        imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))
        if len(imgs) > 10:
            separate = np.linspace(0, len(imgs) - 1, 7, endpoint=True, dtype=np.int16)
            selected = [random.randint(separate[idx], separate[idx + 1]) for idx in range(6)]
            img_array_ls = []
            for idx in selected:
                img_pt = imgs[idx]
                img_array = Image.open(img_pt).convert("RGB")
                img_array_ls.append(img_array)
            return img_array_ls
        else:
            raise ValueError("encountered bad input {}".format(self.img_dir_ls[index]))

    def _get_wav_sample(self, index):
        img_dir_name = os.path.basename(self.img_dir_ls[index])
        audio_name = f"{img_dir_name}.wav_mt.csv"
        wav_path = os.path.join(self.data_root, self.audio_dir, audio_name)
        wav_ft = np.loadtxt(wav_path, delimiter=",")
        return wav_ft


def make_data_loader(cfg, mode):
    assert (mode in ["train", "valid", "test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = face_image_transform()
    if mode == "train":
        dataset = TemporalData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_AUD_DATA,
            cfg.TRAIN_LABEL_DATA,
            transforms
        )
        batch_size = cfg.TRAIN_BATCH_SIZE
    elif mode == "valid":
        dataset = TemporalData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_AUD_DATA,
            cfg.VALID_LABEL_DATA,
            transforms
        )
        batch_size = cfg.VALID_BATCH_SIZE
    else:
        dataset = TemporalData(
            cfg.DATA_ROOT,
            cfg.TEST_IMG_DATA,
            cfg.TEST_AUD_DATA,
            cfg.TEST_LABEL_DATA,
            transforms
        )
        batch_size = cfg.VALID_BATCH_SIZE
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # cfg.NUM_WORKS
        drop_last=True,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def bimodal_lstm_data_loader(cfg, mode):
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg)
    if mode == "train":
        dataset = TemporalData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        dataset = TemporalData(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        return
    else:
        dataset = TemporalData(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # cfg.NUM_WORKS
        drop_last=cfg.DATA_LOADER.DROP_LAST,
    )
    return data_loader


if __name__ == "__main__":
    trans = face_image_transform()
    data_set = TemporalData(
        "../../../datasets",
        "image_data/train_data_face",
        "voice_data/voice_mfcc/train_data_mfcc",
        "annotation/annotation_training.pkl",
        trans
    )
    print(len(data_set))
    for key, val in data_set[7].items():
        print(key, val.shape)
    # print(data_set._statistic_img_sample(1))
    # print(data_set._get_wav_sample(1))
    # loader = make_data_loader("", "train")
    # for i, sample in enumerate(loader):
    #     if i > 0:
    #         break
    #     print(sample["image"].shape, sample["audio"].shape, sample["label"].shape)
