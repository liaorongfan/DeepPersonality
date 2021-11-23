import os
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
        label = self.get_ocean_label(index)
        sample = {
            "aud_data": aud_data,
            "aud_label": label,
        }
        return sample

    def get_wave_data(self, index):
        aud = self.aud_file_ls[index]
        return aud

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
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    return data_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = AudioData(
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