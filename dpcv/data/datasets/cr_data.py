import torch
import os
import glob
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np
from pathlib import Path
from dpcv.data.datasets.bi_modal_data import VideoData
from .build import DATA_LOADER_REGISTRY
from dpcv.data.transforms.transform import set_crnet_transform, crnet_frame_face_transform
from dpcv.data.transforms.build import build_transform_spatial


class CRNetData(VideoData):
    def __init__(self, data_root, img_dir, face_img_dir, audio_dir, label_file, transform=None, sample_size=100):
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size
        self.face_img_dir_ls = self.get_face_img_dir(face_img_dir)

    def get_face_img_dir(self, face_img_dir):
        dir_ls = os.listdir(os.path.join(self.data_root, face_img_dir))
        return dir_ls

    def __getitem__(self, idx):
        glo_img, loc_img, idx = self.get_imgs(idx)  # in case the idx changed
        wav_aud = self.get_wav_aud(idx)
        anno_score = self.get_ocean_label(idx)
        anno_cls_encode = self.cls_encode(anno_score)

        if self.transform:
            glo_img = self.transform["frame"](glo_img)
            loc_img = self.transform["face"](loc_img)
        wav_aud = torch.as_tensor(wav_aud, dtype=glo_img.dtype)
        anno_score = torch.as_tensor(anno_score, dtype=glo_img.dtype)
        anno_cls_encode = torch.as_tensor(anno_cls_encode)

        sample = {
            "glo_img": glo_img, "loc_img": loc_img, "wav_aud": wav_aud,
            "reg_label": anno_score, "cls_label": anno_cls_encode
        }
        return sample

    @staticmethod
    def cls_encode(score):
        index = []
        for v in score:
            if 0 < v < 0.5:
                index.append(0)
            elif 0.5 <= v < 0.6:
                index.append(1)
            elif 0.6 <= v < 0.7:
                index.append(2)
            else:
                index.append(3)
        one_hot_cls = np.eye(4)[index]
        return one_hot_cls

    def get_imgs(self, idx):

        glo_img_dir = self.img_dir_ls[idx]
        if "train" in glo_img_dir:
            loc_img_dir = glo_img_dir.replace("train_data", "train_data_face")
        elif "valid" in glo_img_dir:
            loc_img_dir = glo_img_dir.replace("valid_data", "valid_data_face")
        else:
            loc_img_dir = glo_img_dir.replace("test_data", "test_data_face")
        # in case some video doesn't get aligned face images
        if os.path.basename(loc_img_dir) not in self.face_img_dir_ls:
            return self.get_imgs(idx + 1)
        loc_imgs = glob.glob(loc_img_dir + "/*.jpg")
        loc_imgs = sorted(loc_imgs, key=lambda x: int(Path(x).stem[5:]))
        # according to the paper sample 32 frames per video
        separate = np.linspace(0, len(loc_imgs), self.sample_size, endpoint=False, dtype=np.int16)
        img_index = random.choice(separate)
        try:
            loc_img_pt = loc_imgs[img_index]
        except IndexError:
            loc_img_pt = loc_imgs[0]
        glo_img_pt = self._match_img(loc_img_pt)

        loc_img_arr = Image.open(loc_img_pt).convert("RGB")
        glo_img_arr = Image.open(glo_img_pt).convert("RGB")

        return glo_img_arr, loc_img_arr, idx

    @staticmethod
    def _match_img(loc_img_pt):
        img_dir = os.path.dirname(loc_img_pt).replace("_face", "")
        img_name, _ = os.path.basename(loc_img_pt).split(".")
        img_id = int(img_name.split("_")[-1])
        glo_img_name = "frame_" + str(img_id) + ".jpg"
        glo_img_path = os.path.join(img_dir, glo_img_name)
        if os.path.exists(glo_img_path):
            return glo_img_path
        else:
            return os.path.join(img_dir, "frame_1.jpg")

    def get_wav_aud(self, index):
        img_dir_name = os.path.basename(self.img_dir_ls[index])
        audio_name = f"{img_dir_name}.wav.npy"
        wav_path = os.path.join(self.data_root, self.audio_dir, audio_name)
        wav_ft = np.load(wav_path, allow_pickle=True)
        if wav_ft.shape[-1] < 244832:
            wav_ft_pad = np.zeros((1, 1, 244832))
            wav_ft_pad[..., :wav_ft.shape[-1]] = wav_ft
            return wav_ft_pad
        return wav_ft


class AllFrameCRNetData(CRNetData):

    def __getitem__(self, idx):
        glo_img_ls, loc_img_ls, idx = self.get_imgs(idx)
        wav_aud = self.get_wav_aud(idx)
        anno_score = self.get_ocean_label(idx)

        if self.transform:
            glo_img_ls = [self.transform["frame"](img) for img in glo_img_ls]
            loc_img_ls = [self.transform["face"](img) for img in loc_img_ls]
        wav_aud = torch.as_tensor(wav_aud, dtype=glo_img_ls[0].dtype)
        anno_score = torch.as_tensor(anno_score, dtype=glo_img_ls[0].dtype)

        sample = {
            "glo_img": glo_img_ls,
            "loc_img": loc_img_ls,
            "wav_aud": wav_aud,
            "reg_label": anno_score,
        }
        return sample

    def get_imgs(self, idx):
        glo_img_dir = self.img_dir_ls[idx]
        if "train" in glo_img_dir:
            loc_img_dir = glo_img_dir.replace("train_data", "train_data_face")
        elif "valid" in glo_img_dir:
            loc_img_dir = glo_img_dir.replace("valid_data", "valid_data_face")
        else:
            loc_img_dir = glo_img_dir.replace("test_data", "test_data_face")
        # in case some video doesn't get aligned face images
        if os.path.basename(loc_img_dir) not in self.face_img_dir_ls:
            return self.get_imgs(idx + 1)

        loc_imgs = glob.glob(loc_img_dir + "/*.jpg")
        loc_imgs = sorted(loc_imgs, key=lambda x: int(Path(x).stem[5:]))

        loc_img_ls, glo_img_ls = [], []
        # separate = np.linspace(0, len(loc_imgs), self.sample_size, endpoint=False, dtype=np.int16)
        separate = list(range(len(loc_imgs)))
        for img_index in separate:
            try:
                loc_img_pt = loc_imgs[img_index]
            except IndexError:
                loc_img_pt = loc_imgs[0]

            glo_img_pt = self._match_img(loc_img_pt)

            loc_img_ls.append(loc_img_pt)
            glo_img_ls.append(glo_img_pt)

        loc_img_obj_ls = [Image.open(loc_img) for loc_img in loc_img_ls]
        glo_img_obj_ls = [Image.open(glo_img) for glo_img in glo_img_ls]

        return glo_img_obj_ls, loc_img_obj_ls, idx


def make_data_loader(cfg, mode=None):
    assert (mode in ["train", "valid", "test"]), " 'mode' only supports 'train' and 'valid'"
    transforms = set_crnet_transform()
    if mode == "train":
        dataset = CRNetData(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TRAIN_IMG_DATA,  # "image_data/train_data",
            cfg.TRAIN_IMG_FACE_DATA,  # "image_data/train_data_face",
            cfg.TRAIN_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms
        )
    elif mode == "valid":
        dataset = CRNetData(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.VALID_IMG_DATA,  # "image_data/train_data",
            cfg.VALID_IMG_FACE_DATA,  # "image_data/train_data_face",
            cfg.VALID_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            cfg.VALID_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms
        )
    else:
        dataset = CRNetData(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TEST_IMG_DATA,  # "image_data/train_data",
            cfg.TEST_IMG_FACE_DATA,  # "image_data/train_data_face",
            cfg.TEST_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            cfg.TEST_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,  # cfg.NUM_WORKS
        drop_last=True,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def crnet_data_loader(cfg, mode=None):
    assert (mode in ["train", "valid", "test", "full_test"]), \
        " 'mode' only supports 'train', 'valid', 'test' and 'full_test' "

    transforms = build_transform_spatial(cfg)
    data_cfg = cfg.DATA
    if mode == "train":
        dataset = CRNetData(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.TRAIN_IMG_DATA,  # "image_data/train_data",
            data_cfg.TRAIN_IMG_FACE_DATA,  # "image_data/train_data_face",
            data_cfg.TRAIN_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            data_cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms,
        )
    elif mode == "valid":
        dataset = CRNetData(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.VALID_IMG_DATA,  # "image_data/train_data",
            data_cfg.VALID_IMG_FACE_DATA,  # "image_data/train_data_face",
            data_cfg.VALID_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            data_cfg.VALID_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms,
        )
    elif mode == "full_test":
        return AllFrameCRNetData(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.TEST_IMG_DATA,  # "image_data/train_data",
            data_cfg.TEST_IMG_FACE_DATA,  # "image_data/train_data_face",
            data_cfg.TEST_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            data_cfg.TEST_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms,
        )
    else:
        dataset = CRNetData(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.TEST_IMG_DATA,  # "image_data/train_data",
            data_cfg.TEST_IMG_FACE_DATA,  # "image_data/train_data_face",
            data_cfg.TEST_AUD_DATA,  # "voice_data/train_data",  # default train_data_244832 form librosa
            data_cfg.TEST_LABEL_DATA,  # "annotation/annotation_training.pkl",
            transforms,
        )
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        shuffle=loader_cfg.SHUFFLE,
        num_workers=loader_cfg.NUM_WORKERS,  # cfg.NUM_WORKS
        drop_last=loader_cfg.DROP_LAST,
    )
    return data_loader


if __name__ == "__main__":
    trans = set_crnet_transform()
    data_set = CRNetData(
        "../../../datasets",
        "image_data/train_data",
        "image_data/train_data_face",
        "voice_data/train_data",
        "annotation/annotation_training.pkl",
        trans
    )
    print(len(data_set))
    print(data_set[2])
    # for item in data_set[2].values():
    #     print(item.shape)
    # # print(data_set._statistic_img_sample(1))
    # # print(data_set._get_wav_sample(1))

    # loader = make_data_loader("", mode="train")
    # for i, sample in enumerate(loader):
    #     # if i > 5:
    #     #     break
    #     for item in sample.values():
    #         print(item.shape)
