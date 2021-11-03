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
from data.transforms.transform import set_lstm_transform


class TemporalData(VideoData):
    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None):
        super().__init__(data_root, img_dir, label_file, audio_dir)
        # self.data_root = data_root
        # self.audio_dir = audio_dir
        # self.img_dir_pt = img_dir
        self.transform = transform
        # self.img_dir_ls = self.parse_img_dir(img_dir)  # every directory name indeed a video
        # self.annotation = self.parse_annotation(label_file)

    # def parse_img_dir(self, img_dir):
    #     img_dir_ls = os.listdir(os.path.join(self.data_root, img_dir))
    #     img_dir_ls = [img_dir.replace("_aligned", "") for img_dir in img_dir_ls if "aligned" in img_dir]
    #     return img_dir_ls
    #
    # def parse_annotation(self, label_file):
    #     label_path = os.path.join(self.data_root, label_file)
    #     with open(label_path, "rb") as f:
    #         annotation = pickle.load(f, encoding="latin1")
    #     return annotation
    #
    # def _find_ocean_score(self, index):
    #     video_name = f"{self.img_dir_ls[index]}.mp4"
    #     score = [
    #         self.annotation["openness"][video_name],
    #         self.annotation["conscientiousness"][video_name],
    #         self.annotation["extraversion"][video_name],
    #         self.annotation["agreeableness"][video_name],
    #         self.annotation["neuroticism"][video_name],
    #     ]
    #     return score

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
        # img_dir_name = f"{self.img_dir_ls[index]}_aligned"
        # img_dir_path = os.path.join(self.data_root, self.img_dir_pt, img_dir_name)
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
    transforms = set_lstm_transform()
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


if __name__ == "__main__":
    trans = set_lstm_transform()
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
