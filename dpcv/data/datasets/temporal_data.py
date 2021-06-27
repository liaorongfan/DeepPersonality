import torch
import os
import pickle
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np
from dpcv.data.datasets.transforms import set_transform_op


class TemporalData(Dataset):
    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None):
        self.data_root = data_root
        self.audio_dir = audio_dir
        self.img_dir_pt = img_dir
        self.transform = transform
        self.img_dir_ls = self.parse_img_dir(img_dir)  # every directory name indeed a video
        self.annotation = self.parse_annotation(label_file)

    def parse_img_dir(self, img_dir):
        img_dir_ls = os.listdir(os.path.join(self.data_root, img_dir))
        img_dir_ls = [img_dir.replace("_aligned", "") for img_dir in img_dir_ls if "aligned" in img_dir]
        return img_dir_ls

    def parse_annotation(self, label_file):
        label_path = os.path.join(self.data_root, label_file)
        with open(label_path, "rb") as f:
            annotation = pickle.load(f, encoding="latin1")
        return annotation

    def _find_ocean_score(self, index):
        video_name = self.img_dir_ls[index] + ".mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return score

    def __getitem__(self, idx):
        anno_score = self._find_ocean_score(idx)
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
        img_dir_name = self.img_dir_ls[index] + "_aligned"
        img_dir_path = os.path.join(self.data_root, self.img_dir_pt, img_dir_name)
        imgs = glob.glob(img_dir_path + "/*.bmp")
        separate = [idx for idx in range(0, 100, 16)]  # according to the paper separate the video into 6 sessions
        if len(imgs) < 100:
            imgs_nums = len(imgs)
            separate = [idx for idx in range(0, imgs_nums, int(imgs_nums / 6))]
        selected = [random.randint(separate[idx], separate[idx + 1]) for idx in range(6)]
        img_array_ls = []
        for idx in selected:
            try:
                img_pt = imgs[idx]
                img_array = Image.open(img_pt).convert("RGB")
                img_array_ls.append(img_array)
            except:
                print("encounter bad image:", imgs[idx])
        # video_imgs = np.stack(img_array_ls, axis=0)
        return img_array_ls

    def _get_wav_sample(self, index):
        img_dir_name = self.img_dir_ls[index] + ".wav_mt.csv"
        wav_path = os.path.join(self.data_root, self.audio_dir, img_dir_name)
        wav_ft = np.loadtxt(wav_path, delimiter=",")
        return wav_ft


def make_data_loader(cfg, mode):
    assert (mode in ["train", "valid"]), " 'mode' only supports 'train' and 'valid'"
    transforms = set_transform_op()
    if mode == "train":
        dataset = TemporalData(
            cfg.DATA_ROOT,
            cfg.TRAIN_IMG_DATA,
            cfg.TRAIN_AUD_DATA,
            cfg.TRAIN_LABEL_DATA,
            transforms
        )
        batch_size = cfg.TRAIN_BATCH_SIZE
    else:
        dataset = TemporalData(
            cfg.DATA_ROOT,
            cfg.VALID_IMG_DATA,
            cfg.VALID_AUD_DATA,
            cfg.VALID_LABEL_DATA,
            transforms
        )
        batch_size = cfg.VALID_BATCH_SIZE
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # cfg.NUM_WORKS
    )
    return data_loader


if __name__ == "__main__":
    # trans = set_transform_op()
    # data_set = TemporalData(
    #     "../../../datasets",
    #     "ImageData/trainingData_face",
    #     "VoiceData/trainingData_mfcc",
    #     "annotation_training.pkl",
    #     trans
    # )
    # print(len(data_set))
    # print(data_set[1])
    # print(data_set._statistic_img_sample(1))
    # print(data_set._get_wav_sample(1))
    loader = make_data_loader("", "train")
    for i, sample in enumerate(loader):
        if i > 0:
            break
        print(sample["image"].shape, sample["audio"].shape, sample["label"].shape)
