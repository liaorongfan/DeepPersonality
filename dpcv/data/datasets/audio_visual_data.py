import os
import torch
import glob
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np
from dpcv.data.datasets.bi_modal_data import BimodalData
from dpcv.data.datasets.transforms import set_audio_visual_transform


class AudioVisualData(BimodalData):

    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None):
        super().__init__(data_root, img_dir, audio_dir, label_file)
        self.transform = transform

    def __getitem__(self, idx):
        label = self._find_ocean_score(idx)
        img = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            img = self.transform(img)

        wav = torch.as_tensor(wav, dtype=img.dtype)
        label = torch.as_tensor(label, dtype=img.dtype)

        sample = {"image": img, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir = self.img_dir_ls[idx]
        img_dir_path = os.path.join(self.data_root, self.img_dir, img_dir)
        img_paths = glob.glob(img_dir_path + "/*.jpg")
        img_path = random.choice(img_paths)
        try:
            img = Image.open(img_path).convert("RGB")
            return img
        except:
            print(img_path)

    def get_wave_data(self, idx):
        file_name = f"{self.img_dir_ls[idx]}.wav.npy"
        wav_path = os.path.join(self.data_root, self.audio_dir, file_name)
        wav_ft = np.load(wav_path)
        return wav_ft


# def make_data_loader_(cfg, mode):
#     assert (mode in ["train", "valid"]), " 'mode' only supports 'train' and 'valid'"
#     transforms = set_audio_visual_transform()
#     if mode == "train":
#         dataset = AudioVisualData(
#             cfg.DATA_ROOT,
#             cfg.TRAIN_IMG_DATA,
#             cfg.TRAIN_AUD_DATA,
#             cfg.TRAIN_LABEL_DATA,
#             transforms
#         )
#         batch_size = cfg.TRAIN_BATCH_SIZE
#     else:
#         dataset = AudioVisualData(
#             cfg.DATA_ROOT,
#             cfg.VALID_IMG_DATA,
#             cfg.VALID_AUD_DATA,
#             cfg.VALID_LABEL_DATA,
#             transforms
#         )
#         batch_size = cfg.VALID_BATCH_SIZE
#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0  # cfg.NUM_WORKS
#     )
#     return data_loader

def make_data_loader(cfg, mode):
    trans = set_audio_visual_transform()
    if mode == "train":
        data_set = AudioVisualData(
            "../datasets",
            "ImageData/trainingData",
            "VoiceData/trainingData_50176",
            "annotation_training.pkl",
            trans
        )
    else:
        data_set = AudioVisualData(
            "../datasets",
            "ImageData/validationData",
            "VoiceData/validationData_50176",
            "annotation_validation.pkl",
            trans
        )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    return data_loader


if __name__ == "__main__":
    from tqdm import tqdm
    args = ("../../../datasets", "ImageData/trainingData", "VoiceData/trainingData_50176", "annotation_training.pkl")
    trans = set_audio_visual_transform()
    data_set = AudioVisualData(*args, trans)
    # print(len(data_set))
    data = data_set[1]
    print(data["image"].shape, data["audio"].shape, data["label"].shape)

