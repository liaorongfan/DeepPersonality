import os
import torch
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import random
import pickle
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_audio_visual_transform
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
from random import shuffle


class AudioVisualData(VideoData):

    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=100):
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size

    def __getitem__(self, idx):
        label = self.get_ocean_label(idx)
        img = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            img = self.transform(img)

        wav = torch.as_tensor(wav, dtype=img.dtype)
        label = torch.as_tensor(label, dtype=img.dtype)

        sample = {"image": img, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir_path = self.img_dir_ls[idx]
        img_paths = glob.glob(img_dir_path + "/*.jpg")
        sample_frames = np.linspace(0, len(img_paths), self.sample_size, endpoint=False, dtype=np.int16)
        selected = random.choice(sample_frames)
        # img_path = random.choice(img_paths)
        try:
            img = Image.open(img_paths[selected]).convert("RGB")
            return img
        except:
            print(img_paths)

    def get_wave_data(self, idx):

        img_dir_path = self.img_dir_ls[idx]
        # wav_path = img_dir_path.replace("image_data", "voice_data/voice_librosa") + ".wav.npy"
        video_name = os.path.basename(img_dir_path)
        wav_path = os.path.join(self.data_root, self.audio_dir, f"{video_name}.wav.npy")
        wav_ft = np.load(wav_path)
        try:
            n = np.random.randint(0, len(wav_ft) - 50176)
        except:
            n = 0
        wav_tmp = wav_ft[..., n: n + 50176]
        if wav_tmp.shape[-1] < 50176:
            wav_fill = np.zeros((1, 1, 50176))
            wav_fill[..., :wav_tmp.shape[-1]] = wav_tmp
            wav_tmp = wav_fill
        return wav_tmp


class ALLSampleAudioVisualData(AudioVisualData):

    def __getitem__(self, idx):
        label = self.get_ocean_label(idx)
        imgs = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        wav = torch.as_tensor(wav, dtype=imgs[0].dtype)
        label = torch.as_tensor(label, dtype=imgs[0].dtype)

        sample = {"image": imgs, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir_path = self.img_dir_ls[idx]
        img_path_ls = glob.glob(f"{img_dir_path}/*.jpg")
        sample_frames = np.linspace(0, len(img_path_ls), self.sample_size, endpoint=False, dtype=np.int16)
        img_path_ls_sampled = [img_path_ls[ind] for ind in sample_frames]
        img_obj_ls = [Image.open(path) for path in img_path_ls_sampled]
        return img_obj_ls


class ALLSampleAudioVisualData2(AudioVisualData):

    def __getitem__(self, idx):
        label = self.get_ocean_label(idx)
        imgs = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        wav = torch.as_tensor(wav, dtype=imgs[0].dtype)
        label = torch.as_tensor(label, dtype=imgs[0].dtype)

        sample = {"image": imgs, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir_path = self.img_dir_ls[idx]
        img_path_ls = sorted(glob.glob(f"{img_dir_path}/*.jpg"))
        # sample_frames = np.linspace(0, len(img_path_ls), self.sample_size, endpoint=False, dtype=np.int16)
        # img_path_ls_sampled = [img_path_ls[ind] for ind in sample_frames]
        img_obj_ls = [Image.open(path) for path in img_path_ls]
        return img_obj_ls


def make_data_loader(cfg, mode):
    trans = set_audio_visual_transform()
    if mode == "train":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.TRAIN_IMG_DATA,  # "image_data/train_data",
            cfg.TRAIN_AUD_DATA,  # "voice_data/train_data",
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            trans
        )
    elif mode == "valid":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.VALID_IMG_DATA,  # "image_data/valid_data",
            cfg.VALID_AUD_DATA,  # "voice_data/valid_data",
            cfg.VALID_LABEL_DATA,  # annotation/annotation_validation.pkl",
            trans
        )
    elif mode == "trainval":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TRAINVAL_IMG_DATA,  # ["image_data/training_data_01", "image_data/validation_data_01"],
            cfg.TRANIVAL_AUD_DATA,  # ["voice_data/trainingData", "voice_data/validationData"],
            cfg.TRAINVAL_LABEL_DATA,  # ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
            trans,
        )
    elif mode == "test":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.TEST_IMG_DATA,  # "image_data/test_data",
            cfg.TEST_AUD_DATA,  # "voice_data/test_data",
            cfg.TEST_LABEL_DATA,  # "annotation/annotation_test.pkl",
            trans
        )
    else:
        raise ValueError("mode must in one of [train, valid, trianval, test]")

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKS,
        drop_last=True
    )

    return data_loader


@DATA_LOADER_REGISTRY.register()
def bimodal_resnet_data_loader(cfg, mode):
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg)
    if mode == "train":
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        return ALLSampleAudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transforms
        )
    else:
        dataset = AudioVisualData(
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
    # from tqdm import tqdm
    # args = ("../../../datasets", "ImageData/trainingData", "VoiceData/trainingData_50176", "annotation_training.pkl")
    trans = set_audio_visual_transform()
    # data_set = AudioVisualData(*args, trans)
    # # print(len(data_set))
    # data = data_set[1]
    # print(data["image"].shape, data["audio"].shape, data["label"].shape)

    dataset = AudioVisualData(
        "../../../datasets",
        ["image_data/training_data_01", "image_data/validation_data_01"],
        ["voice_data/trainingData", "voice_data/validationData"],
        ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
        trans,
    )
    print(len(dataset))
    a = dataset[1]
    print(a)
