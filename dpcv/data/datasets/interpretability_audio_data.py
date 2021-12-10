import os
from dpcv.data.datasets.bi_modal_data import VideoData
from .build import DATA_LOADER_REGISTRY
import torch
import torchaudio
from torch.utils.data import DataLoader


def aud_transform():
    return torchaudio.transforms.Resample(44100, 4000)


def norm(aud_ten):
    mean = aud_ten.mean()
    std = aud_ten.std()
    normed_ten = (aud_ten - mean) / (std + 1e-10)
    return normed_ten


class InterpretAudio(VideoData):
    def __init__(self, data_root, aud_dir, label_file):
        super().__init__(
            data_root, img_dir=None, audio_dir=aud_dir, label_file=label_file,
            parse_img_dir=False,
            parse_aud_dir=True,
        )

    def __getitem__(self, index):
        aud_data, index = self.get_wave_data(index)
        label = self.get_ocean_label(index)
        sample = {
            "aud_data": aud_data,
            "aud_label": label,
        }
        return sample

    def get_wave_data(self, index):
        aud_file = self.aud_file_ls[index]
        aud_data, sample_rate = torchaudio.load(aud_file)
        trans_aud = torchaudio.transforms.Resample(sample_rate, 4000)(aud_data[0, :].view(1, -1))
        trans_fft = torch.fft.fft(trans_aud)
        half_length = int(trans_aud.shape[-1] / 2)
        trans_fre = torch.abs(trans_fft)[..., :half_length]
        trans_fre_norm = norm(trans_fre)
        if trans_fre_norm.shape[-1] < 30604:
            return self.get_wave_data(index - 1)
        return trans_fre_norm, index

    def get_ocean_label(self, index):
        aud_file = self.aud_file_ls[index]
        video_name = os.path.basename(aud_file).replace(".wav", ".mp4")
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return torch.tensor(score)

    def __len__(self):
        return len(self.aud_file_ls)


def make_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = InterpretAudio(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TRAIN_AUD_DATA,  # "raw_voice/trainingData",
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
        )
    elif mode == "valid":
        dataset = InterpretAudio(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.VALID_AUD_DATA,  # "raw_voice/validationData",
            cfg.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    elif mode == "test":
        dataset = InterpretAudio(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TEST_AUD_DATA,  # "raw_voice/testData",
            cfg.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    else:
        raise ValueError("mode must be one of 'train' or 'valid' or test' ")

    data_loader = DataLoader(dataset, batch_size=128, num_workers=cfg.NUM_WORKERS)

    return data_loader


@DATA_LOADER_REGISTRY.register()
def interpret_audio_dataloader(cfg, mode):
    data_cfg = cfg.DATA
    if mode == "train":
        dataset = InterpretAudio(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.TRAIN_AUD_DATA,  # "raw_voice/trainingData",
            data_cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
        )
    elif mode == "valid":
        dataset = InterpretAudio(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.VALID_AUD_DATA,  # "raw_voice/validationData",
            data_cfg.VALID_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    elif mode == "test":
        dataset = InterpretAudio(
            data_cfg.ROOT,  # "../datasets",
            data_cfg.TEST_AUD_DATA,  # "raw_voice/testData",
            data_cfg.TEST_LABEL_DATA,  # "annotation/annotation_validation.pkl",
        )
    else:
        raise ValueError("mode must be one of 'train' or 'valid' or test' ")
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        num_workers=loader_cfg.NUM_WORKERS,
    )

    return data_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = InterpretAudio(
        "../../../datasets",
        "raw_voice/trainingData",
        "annotation/annotation_training.pkl",
    )

    data_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for i, batch in enumerate(data_loader):
        if i >= 20:
            break
        plt.plot(batch[0].squeeze())
        plt.show()
        print(batch[0].shape, batch[1].shape)
