import os
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData
from torch.utils.data import DataLoader
from .build import DATA_LOADER_REGISTRY
import torch


class AudioData(VideoData):
    def __init__(self, data_root, aud_dir, label_file):
        super().__init__(
            data_root, img_dir=None, audio_dir=aud_dir, label_file=label_file,
            parse_img_dir=False,
            parse_aud_dir=True,
        )

    def __getitem__(self, index):
        aud_data = self.get_wave_data(index)
        aud_data = self.transform(aud_data)
        label = self.get_ocean_label(index)
        sample = {
            "aud_data": aud_data,
            "aud_label": label,
        }
        return sample

    def get_wave_data(self, index):
        aud_file = self.aud_file_ls[index]
        aud_ft = np.load(aud_file)
        return aud_ft

    def get_ocean_label(self, index):
        aud_file = self.aud_file_ls[index]
        aud_name = os.path.basename(aud_file)
        video_name = aud_name.replace(".wav", "").replace(".npy", "").replace(".wav_mt.csv", "") + ".mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return torch.tensor(score)

    def transform(self, aud_ft):
        """
        interface to be override for aud data processing
        """
        return aud_ft

    def __len__(self):
        return len(self.aud_file_ls)


class VoiceLibrosa(AudioData):

    def transform(self, aud_ft):
        _, _, length = aud_ft.shape
        aud_padding = np.zeros((1, 1, 245760))
        aud_padding[..., :length] = aud_ft
        aud_trans = aud_padding.reshape(256, 320, 3).transpose(2, 0, 1)
        aud_ts = torch.as_tensor(aud_trans, dtype=torch.float32)
        return aud_ts


@DATA_LOADER_REGISTRY.register()
def build_audio_loader(cfg, mode="train"):
    if mode == "train":
        dataset = AudioData(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.TRAIN_AUD_DATA,  # "raw_voice/trainingData",
            cfg.DATA.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
        )
    elif mode == "valid":
        dataset = AudioData(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.VALID_AUD_DATA,  # "raw_voice/validationData",
            cfg.DATA.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    elif mode == "test":
        dataset = AudioData(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.TEST_AUD_DATA,  # "raw_voice/testData",
            cfg.DATA.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    else:
        raise ValueError("mode must be one of 'train' or 'valid' or test' ")

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    return data_loader


@DATA_LOADER_REGISTRY.register()
def voice_librosa_loader(cfg, mode="train"):
    if mode == "train":
        dataset = VoiceLibrosa(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.TRAIN_AUD_DATA,  # "raw_voice/trainingData",
            cfg.DATA.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
        )
    elif mode == "valid":
        dataset = VoiceLibrosa(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.VALID_AUD_DATA,  # "raw_voice/validationData",
            cfg.DATA.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    elif mode == "test":
        dataset = VoiceLibrosa(
            cfg.DATA.ROOT,  # "../datasets",
            cfg.DATA.TEST_AUD_DATA,  # "raw_voice/testData",
            cfg.DATA.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    else:
        raise ValueError("mode must be one of 'train' or 'valid' or test' ")

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    return data_loader


if __name__ == "__main__":

    dataset = VoiceLibrosa(
        "../../../datasets",
        "voice_data/voice_librosa/train_data",
        "annotation/annotation_training.pkl",
    )
    # for i in range(len(dataset)):
    #     a = dataset[i]
    data_loader = DataLoader(dataset, batch_size=8, num_workers=0)
    for i, batch in enumerate(data_loader):
        print(batch["aud_data"].shape, batch["aud_label"].shape)
        if i >= 20:
            break
    #     plt.plot(batch[0].squeeze())
    #     plt.show()
    #     print(batch[0].shape, batch[1].shape)
