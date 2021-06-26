import os
import torch
import glob
from torch.utils.data import DataLoader
import librosa
from PIL import Image
import random
import numpy as np
from dpcv.data.datasets.chalearn_data import BimodalData
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
        img = Image.open(img_path).convert("RGB")
        return img

    def get_wave_data(self, idx):
        img_dir_name = self.img_dir_ls[idx] + ".wav"
        wav_path = os.path.join(self.data_root, self.audio_dir, img_dir_name)
        wav_ft = librosa.load(wav_path, 3279)[0][None, None, :]
        if wav_ft.shape[-1] < 50176:
            wav_temp = np.zeros((1, 1, 50176))
            wav_temp[..., :wav_ft.shape[-1]] = wav_ft
            return wav_temp
        return wav_ft


def make_data_loader():
    trans = set_audio_visual_transform()
    data_set = AudioVisualData(
        "../datasets",
        "ImageData/trainingData",
        "VoiceData/trainingData",
        "annotation_training.pkl",
        trans
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    return data_loader


if __name__ == "__main__":
    trans = set_audio_visual_transform()
    data_set = AudioVisualData(
        "../../../datasets",
        "ImageData/trainingData",
        "VoiceData/trainingData",
        "annotation_training.pkl",
        trans
    )
    print(len(data_set))
    data = data_set[1]
    print(data["image"].shape, data["audio"].shape, data["label"].shape)
