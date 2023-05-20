from .ture_personality_data import Chalearn21FrameData, Chalearn21PersemonData
from dpcv.data.transforms.build import build_transform_spatial
import glob
import torch
import os.path as opt
from pathlib import Path
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AllSampleTruePersonalityData(Chalearn21FrameData):

    def __init__(
        self, data_root, data_split, task,
        data_type="frame", img_format="jpg", even_downsample=1000, trans=None, segment=True,
    ):
        super().__init__(
            data_root, data_split, task, data_type, even_downsample, trans, segment,
        )
        self.img_format = img_format

    def __len__(self):
        return len(self.img_dir_ls)

    def __getitem__(self, idx):
        img_obj_ls, img_file_ls = self.get_sample_frames(idx)
        img_label = self.get_ocean_label(img_file_ls[0])
        if self.trans:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        return {"all_images": img_obj_ls, "label": torch.as_tensor(img_label, dtype=torch.float32)}

    def get_sample_frames(self, idx):
        img_dir = self.img_dir_ls[idx]
        imgs = glob.glob(opt.join(self.data_dir, img_dir, f"*.{self.img_format}"))
        if self.type == "frame":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[6:]))
        elif self.type == "face":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))
        if self.sample_size:
            separate = np.linspace(0, len(imgs), self.sample_size, endpoint=False, dtype=np.int32)
            imgs = [imgs[idx] for idx in separate]
        img_obj_ls = [Image.open(img_path) for img_path in imgs]
        return img_obj_ls, imgs


class AllSampleBimodalTruePersonalityData(AllSampleTruePersonalityData):
    sample_len = 50176

    def __getitem__(self, idx):
        img_obj_ls, img_file_ls = self.get_sample_frames(idx)
        img_label = self.get_ocean_label(img_file_ls[0])
        wav = self.get_wave_data(img_file_ls[0])
        if self.trans:
            img_obj_ls = [self.trans(img) for img in img_obj_ls]
        wav = torch.as_tensor(wav, dtype=torch.float32)
        img_label = torch.as_tensor(img_label, dtype=torch.float32)
        sample = {"image": img_obj_ls, "audio": wav, "label": img_label}
        return sample

    def get_wave_data(self, file_name):
        dir_name = opt.dirname(file_name)
        if self.type == "frame":
            aud_file = f"{dir_name}.npy"
        if self.type == "face":
            dir_name = dir_name.replace("_face", "")
            aud_file = f"{dir_name}.npy"
        aud_data = np.load(aud_file)
        data_len = aud_data.shape[-1]
        start = np.random.randint(data_len - self.sample_len)
        end = start + self.sample_len
        return aud_data[:, :, start: end]


class AllSamplePersemonTruePersonalityData(Chalearn21PersemonData):

    def __init__(
            self, data_root, data_split, task, data_type, trans,
            emo_data_root, emo_img_dir, emo_label, emo_trans,
    ):
        super().__init__(
            data_root, data_split, task, data_type, trans,
            emo_data_root, emo_img_dir, emo_label, emo_trans, segment=True,
        )
        self.emo_data_root = emo_data_root
        self.emo_img_dir = emo_img_dir
        self.emo_label = emo_label
        self.emo_trans = emo_trans
        self.emo_data_ls = self.emo_data_parser()

    def __getitem__(self, idx):
        per_img_ls, img_file_ls = self.get_sample_frames(idx)
        per_lab_ls = [self.get_ocean_label(img_file_ls[0])] * len(per_img_ls)
        emo_img_ls, emo_lab_ls = self.gather_emotion_data()
        if self.trans:
            per_img_ls = [self.trans(img) for img in per_img_ls]
        if self.emo_trans:
            emo_img_ls = [self.emo_trans(emo_img) for emo_img in emo_img_ls]

        per_imgs_ts = torch.stack(per_img_ls, 0)
        per_labs = torch.as_tensor(per_lab_ls, dtype=torch.float32)
        emo_imgs_ts = torch.stack(emo_img_ls, 0)
        emo_labs = torch.as_tensor(emo_lab_ls)
        sample = {
            "per_img": per_imgs_ts,
            "emo_img": emo_imgs_ts,
            "per_label": per_labs,
            "emo_label": emo_labs,
        }
        return sample

    def get_sample_frames(self, idx):
        img_dir = self.img_dir_ls[idx]
        imgs = glob.glob(opt.join(self.data_dir, img_dir, f"*.jpg"))
        if self.type == "frame":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[6:]))
        elif self.type == "face":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))
        if self.sample_size:
            separate = np.linspace(0, len(imgs), self.sample_size, endpoint=False, dtype=np.int32)
            imgs = [imgs[idx] for idx in separate]
        img_obj_ls = [Image.open(img_path) for img_path in imgs]
        return img_obj_ls, imgs


class AllSampleCRNetTruePersonalityData(AllSampleTruePersonalityData):
    sample_len = 244832

    def __getitem__(self, idx):
        img_obj_ls, img_file_ls = self.get_sample_frames(idx)
        loc_img_ls = [self.get_loc_img(img_file) for img_file in img_file_ls]

        wav = self.get_wave_data(img_file_ls[0])
        wav = torch.as_tensor(wav, dtype=torch.float32)

        img_label = self.get_ocean_label(img_file_ls[0])
        img_label = torch.as_tensor(img_label, dtype=torch.float32)
        # label_cls_encode = self.cls_encode(img_label)
        if self.trans:
            img_obj_ls = [self.trans["frame"](img) for img in img_obj_ls]
            loc_img_ls = [self.trans["face"](img) for img in loc_img_ls]

        sample = {
            "glo_img": img_obj_ls,
            "loc_img": loc_img_ls,
            "wav_aud": wav,
            "reg_label": img_label,
        }
        return sample

    def get_wave_data(self, img_file):
        dir_name = opt.dirname(img_file)
        if self.type == "frame":
            aud_file = f"{dir_name}.npy"
        if self.type == "face":
            dir_name = dir_name.replace("_face", "")
            aud_file = f"{dir_name}.npy"
        aud_data = np.load(aud_file)
        data_len = aud_data.shape[-1]
        start = np.random.randint(data_len - self.sample_len)
        end = start + self.sample_len
        return aud_data[:, :, start: end]

    @staticmethod
    def get_loc_img(img_file):
        img_file = Path(img_file)
        img_id = img_file.stem.split("_")[-1]
        loc_img_dir = f"{img_file.parent}_face"
        loc_img_file = f"{loc_img_dir}/face_{img_id}.jpg"
        try:
            loc_img = Image.open(loc_img_file)
        except FileNotFoundError:
            loc_img_ls = list(Path(loc_img_dir).rglob("*.jpg"))
            loc_img_file = random.choice(loc_img_ls)
            loc_img = Image.open(loc_img_file)
        return loc_img

    @staticmethod
    def cls_encode(score):
        index = []
        for v in score:
            if v < -1:
                index.append(0)
            elif -1 <= v < 0:
                index.append(1)
            elif 0 <= v < 1:
                index.append(2)
            else:
                index.append(3)
        one_hot_cls = np.eye(4)[index]
        return one_hot_cls


def set_true_personality_dataloader(cfg, mode):
    transform = build_transform_spatial(cfg)
    data_set = AllSampleTruePersonalityData(
        data_root=cfg.DATA.ROOT,
        data_split=mode,
        task=cfg.DATA.SESSION,
        trans=transform,
    )
    return data_set


def set_audiovisual_true_personality_dataloader(cfg, mode):
    transform = build_transform_spatial(cfg)
    data_set = AllSampleBimodalTruePersonalityData(
        data_root=cfg.DATA.ROOT,
        data_split=mode,
        task=cfg.DATA.SESSION,
        trans=transform,
    )
    return data_set


def set_crnet_true_personality_dataloader(cfg, mode):
    transform = build_transform_spatial(cfg)
    data_set = AllSampleCRNetTruePersonalityData(
        data_root=cfg.DATA.ROOT,
        data_split=mode,
        task=cfg.DATA.SESSION,
        trans=transform,
    )
    return data_set


def set_persemon_true_personality_dataloader(cfg, mode):

    transforms = build_transform_spatial(cfg)
    persemon_dataset = AllSamplePersemonTruePersonalityData(
        data_root=cfg.DATA.ROOT,    # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,
        data_type=cfg.DATA.TYPE,  # "frame",
        trans=transforms,
        emo_data_root=cfg.DATA.VA_ROOT,  # "datasets",
        emo_img_dir=cfg.DATA.VA_DATA,  # "va_data/cropped_aligned",
        emo_label=cfg.DATA.VA_TRAIN_LABEL if mode == "train" else cfg.DATA.VA_VALID_LABEL,
        emo_trans=transforms,
    )

    return persemon_dataset


