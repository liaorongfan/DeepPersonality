import glob
import random
import os
from PIL import Image
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.transforms.transform import set_per_transform
from .build import DATA_LOADER_REGISTRY


class PersEmoNData(VideoData):
    def __init__(self, data_root, per_img_dir, per_label, emo_img_dir, emo_label, per_trans=None, emo_trans=None):
        super().__init__(data_root, per_img_dir, per_label)
        self.emo_img_dir = emo_img_dir
        self.emo_label = emo_label
        self.emo_data_ls = self.emo_data_parser()
        self.per_trans = per_trans
        self.emo_trans = emo_trans

    def __getitem__(self, index):
        per_img_ls, per_lab_ls = self.gather_personality_data(index)
        emo_img_ls, emo_lab_ls = self.gather_emotion_data()
        if self.per_trans:
            per_img_ls = [self.per_trans(per_img) for per_img in per_img_ls]
        if self.emo_trans:
            emo_img_ls = [self.emo_trans(emo_img) for emo_img in emo_img_ls]

        per_imgs_ts = torch.stack(per_img_ls, 0)
        per_labs = torch.as_tensor(per_lab_ls)
        emo_imgs_ts = torch.stack(emo_img_ls, 0)
        emo_labs = torch.as_tensor(emo_lab_ls)
        sample = {
            "per_img": per_imgs_ts,
            "emo_img": emo_imgs_ts,
            "per_label": per_labs,
            "emo_label": emo_labs,
        }
        return sample

    def gather_personality_data(self, index):
        img_dirs = self.img_dir_ls[index * 10: (index + 1) * 10]
        img_ls = []
        label_ls = []
        for img_dir in img_dirs:
            imgs, labs = self.per_img_sample(img_dir)
            img_ls.extend(imgs)
            label_ls.extend(labs)
        return img_ls, label_ls

    def per_img_sample(self, img_dir):
        imgs = glob.glob(f"{img_dir}/*.jpg")

        imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))
        separate = np.linspace(0, len(imgs), 11, endpoint=False, dtype=np.int16)
        imgs_idx = [random.randint(separate[idx], separate[idx + 1]) for idx in range(10)]
        imgs = [imgs[idx] for idx in imgs_idx]
        imgs = [Image.open(img) for img in imgs]
        labs = [self.get_per_label(img_dir)] * 10
        return imgs, labs

    def get_per_label(self, img_dir):
        video_name = f"{os.path.basename(img_dir)}.mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return score

    def gather_emotion_data(self):
        file = random.choice(self.emo_data_ls)
        file_name = Path(file).stem
        img_dir = os.path.join(self.data_root, self.emo_img_dir, file_name)
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        imgs = imgs[:100]
        imgs_pt = [os.path.join(img_dir, img) for img in imgs]
        with open(file, 'r') as f:
            frame_label = [map(lambda x: float(x), line.strip().split(",")) for line in f.readlines()[1:]]
        try:
            imgs_label = [list(frame_label[int(img_name.split(".")[0])]) for img_name in imgs]
        except:
            return self.gather_emotion_data()
        imgs_rgb = [Image.open(img_pt) for img_pt in imgs_pt]
        return imgs_rgb, imgs_label

    def emo_data_parser(self):
        emo_label_path = os.path.join(self.data_root, self.emo_label)
        video_files = [file for file in os.listdir(emo_label_path) if len(file) < 12]
        video_files_pt = [os.path.join(emo_label_path, video_file) for video_file in video_files]
        return video_files_pt

    def __len__(self):
        return int(len(self.img_dir_ls) / 10)  # for each mini-batch the author selected 10 videos in chalearn data


class AllFramePersEmoNData(PersEmoNData):

    def gather_personality_data(self, index):
        img_dir = self.img_dir_ls[index]
        img_ls, label_ls = self.per_img_sample(img_dir)
        # assert len(img_ls) == 100, f"image sample from{img_dir} is not enough"
        # print(len(img_ls))
        return img_ls, label_ls

    def per_img_sample(self, img_dir):
        imgs = glob.glob(f"{img_dir}/*.jpg")
        selected_idx = np.linspace(0, len(imgs), 100, endpoint=False, dtype=np.int16)
        selected_img_ls = [imgs[idx] for idx in selected_idx]
        selected_img_obj = [Image.open(img) for img in selected_img_ls]
        selected_img_lab = [self.get_per_label(img_dir)] * 100
        return selected_img_obj, selected_img_lab

    def __len__(self):
        return len(self.img_dir_ls)


class AllFramePersEmoNData2(AllFramePersEmoNData):

    def per_img_sample(self, img_dir):
        imgs = sorted(glob.glob(f"{img_dir}/*.jpg"))
        # selected_idx = np.linspace(0, len(imgs), 100, endpoint=False, dtype=np.int16)
        # selected_img_ls = [imgs[idx] for idx in selected_idx]
        selected_img_obj = [Image.open(img) for img in imgs]
        selected_img_lab = [self.get_per_label(img_dir)] * len(imgs)
        return selected_img_obj, selected_img_lab



def make_data_loader(cfg, mode=None):
    per_trans = set_per_transform()
    emo_trans = set_per_transform()
    if mode == "train":
        dataset = PersEmoNData(
            cfg.DATA_ROOT,  # "../datasets/",
            cfg.TRAIN_IMG_DATA,  # "image_data/train_data_face",
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            cfg.VA_DATA,  # "va_data/cropped_aligned",
            cfg.VA_TRAIN_LABEL,  # "va_data/va_label/VA_Set/Train_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    elif mode == "valid":
        dataset = PersEmoNData(
            cfg.DATA_ROOT,  # "../datasets/",
            cfg.VALID_IMG_DATA,  # "image_data/valid_data_face",
            cfg.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
            cfg.VA_DATA,  # "va_data/cropped_aligned",
            cfg.VA_VALID_LABEL,  # "va_data/va_label/VA_Set/Validation_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    else:
        dataset = PersEmoNData(
            cfg.DATA_ROOT,  # "../datasets/",
            cfg.TEST_IMG_DATA,  # "image_data/valid_data_face",
            cfg.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
            cfg.VA_DATA,  # "va_data/cropped_aligned",
            cfg.VA_VALID_LABEL,  # "va_data/va_label/VA_Set/Validation_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=cfg.SHUFFLE,
        num_workers=cfg.NUM_WORKS
    )

    return data_loader


@DATA_LOADER_REGISTRY.register()
def peremon_data_loader(cfg, mode=None):
    per_trans = build_transform_spatial(cfg)
    emo_trans = build_transform_spatial(cfg)
    data_cfg = cfg.DATA
    if mode == "train":
        dataset = PersEmoNData(
            data_cfg.ROOT,  # "../datasets/",
            data_cfg.TRAIN_IMG_DATA,  # "image_data/train_data_face",
            data_cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            data_cfg.VA_DATA,  # "va_data/cropped_aligned",
            data_cfg.VA_TRAIN_LABEL,  # "va_data/va_label/VA_Set/Train_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    elif mode == "valid":
        dataset = PersEmoNData(
            data_cfg.ROOT,  # "../datasets/",
            data_cfg.VALID_IMG_DATA,  # "image_data/valid_data_face",
            data_cfg.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
            data_cfg.VA_DATA,  # "va_data/cropped_aligned",
            data_cfg.VA_VALID_LABEL,  # "va_data/va_label/VA_Set/Validation_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    elif mode == "full_test":
        return AllFramePersEmoNData(
            data_cfg.ROOT,  # "../datasets/",
            data_cfg.TEST_IMG_DATA,  # "image_data/valid_data_face",
            data_cfg.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
            data_cfg.VA_DATA,  # "va_data/cropped_aligned",
            data_cfg.VA_VALID_LABEL,  # "va_data/va_label/VA_Set/Validation_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    else:
        dataset = PersEmoNData(
            data_cfg.ROOT,  # "../datasets/",
            data_cfg.TEST_IMG_DATA,  # "image_data/valid_data_face",
            data_cfg.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
            data_cfg.VA_DATA,  # "va_data/cropped_aligned",
            data_cfg.VA_VALID_LABEL,  # "va_data/va_label/VA_Set/Validation_Set",
            per_trans=per_trans,
            emo_trans=emo_trans,
        )

    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=loader_cfg.SHUFFLE,
        num_workers=loader_cfg.NUM_WORKERS
    )

    return data_loader


if __name__ == "__main__":
    per_trans = set_per_transform()
    emo_trans = set_per_transform()
    # dataset = PersEmoNData(
    #     "../../../datasets/",
    #     "image_data/train_data_face",
    #     "annotation/annotation_training.pkl",
    #     "va_data/cropped_aligned",
    #     "va_data/va_label/VA_Set/Train_Set",
    #     per_trans=per_trans,
    #     emo_trans=emo_trans,
    # )
    dataset = PersEmoNData(
        "../../../datasets/",
        "image_data/valid_data_face",
        "annotation/annotation_validation.pkl",
        "va_data/cropped_aligned",
        "va_data/va_label/VA_Set/Validation_Set",
        per_trans=per_trans,
        emo_trans=emo_trans,
    )
    for k, v in dataset[2].items():
        print(v.shape)

    # data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=0  # cfg.NUM_WORKS
    # )
    #
    # for i, item in enumerate(data_loader):
    #     # if i >= 1:
    #     #     break
    #     print("::--------------> ", i)
    #     for k, v in item.items():
    #         print(k, v.squeeze().shape)
