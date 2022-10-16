import os
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.datasets.cr_data import CRNetData
from torch.utils.data import DataLoader
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
import torch


@DATA_LOADER_REGISTRY.register()
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
        video_name = aud_name.replace(".wav", "").replace(".npy", "").replace("_mt.csv", "") + ".mp4"
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


@DATA_LOADER_REGISTRY.register()
class VoiceLogfbank(AudioData):

    def transform(self, aud_ft):
        _, length = aud_ft.shape
        if length > 79534:
            aud_trans = aud_ft[..., :79534]
        elif length < 79534:
            aud_padding = np.zeros((1, 79534))
            aud_padding[..., :length] = aud_ft
            aud_trans = aud_padding
        else:
            aud_trans = aud_ft
        return torch.as_tensor(aud_trans, dtype=torch.float32).squeeze()


@DATA_LOADER_REGISTRY.register()
class VoiceMfcc(AudioData):

    def get_wave_data(self, index):
        aud_file = self.aud_file_ls[index]
        aud_ft = np.loadtxt(aud_file, delimiter=",")
        return aud_ft

    def transform(self, aud_ft):
        return torch.as_tensor(aud_ft, dtype=torch.float32)


@DATA_LOADER_REGISTRY.register()
class VoiceLibrosa(AudioData):

    def transform(self, aud_ft):
        try:
            n = np.random.randint(0, len(aud_ft) - 50176)
        except:
            n = 0
        wav_tmp = aud_ft[..., n: n + 50176]
        if wav_tmp.shape[-1] < 50176:
            wav_fill = np.zeros((1, 1, 50176))
            wav_fill[..., :wav_tmp.shape[-1]] = wav_tmp
            wav_tmp = wav_fill
        return torch.as_tensor(wav_tmp, dtype=torch.float32)


@DATA_LOADER_REGISTRY.register()
class VoiceCRNetData(AudioData):

    def __getitem__(self, index):
        aud_data = self.get_wave_data(index)
        aud_data = self.transform(aud_data)
        label = self.get_ocean_label(index)
        label_cls = torch.as_tensor(CRNetData.cls_encode(label), dtype=torch.float32)
        return {
            "aud_data": aud_data,
            "aud_label": label,
            "aud_label_cls": label_cls,
        }

    def transform(self, aud_ft):
        if aud_ft.shape[-1] < 244832:
            aud_ft_pad = np.zeros((1, 1, 244832))
            aud_ft_pad[..., :aud_ft.shape[-1]] = aud_ft
            aud_ft = aud_ft_pad
        return torch.as_tensor(aud_ft, dtype=torch.float32)


class _VoiceLibrosa(AudioData):

    def transform(self, aud_ft):
        _, _, length = aud_ft.shape
        aud_padding = np.zeros((1, 1, 245760))
        aud_padding[..., :length] = aud_ft
        aud_trans = aud_padding.reshape(256, 320, 3).transpose(2, 0, 1)
        aud_ts = torch.as_tensor(aud_trans, dtype=torch.float32)
        return aud_ts


class VoiceLibrosaSwinTransformer(AudioData):

    def transform(self, aud_ft):
        _, _, length = aud_ft.shape
        shape_size = 224 * 224 * 3
        if length < shape_size:
            aud_padding = np.zeros((1, 1, shape_size))
            aud_padding[..., :length] = aud_ft
            aud_ft = aud_padding
        aud_trans = aud_ft[..., :shape_size].reshape(224, 224, 3).transpose(2, 0, 1)
        aud_ts = torch.as_tensor(aud_trans, dtype=torch.float32)
        return aud_ts


@DATA_LOADER_REGISTRY.register()
def build_audio_loader(cfg, dataset_cls, mode="train"):
    shuffle = cfg.DATA_LOADER.SHUFFLE
    if mode == "train":
        dataset = dataset_cls(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
        )
    elif mode == "valid":
        dataset = dataset_cls(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
        )
        shuffle = False
    elif mode == "test":
        dataset = dataset_cls(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
        )
        shuffle = False
    else:
        raise ValueError("mode must be one of 'train' or 'valid' or test' ")

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        drop_last=cfg.DATA_LOADER.DROP_LAST,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )

    return data_loader


@DATA_LOADER_REGISTRY.register()
def voice_librosa_loader(cfg, mode="train"):
    if mode == "train":
        dataset = _VoiceLibrosa(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
        )
    elif mode == "valid":
        dataset = _VoiceLibrosa(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
        )
    elif mode == "test":
        dataset = _VoiceLibrosa(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
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
def voice_librosa_swin_transformer_loader(cfg, mode="train"):
    if mode == "train":
        dataset = VoiceLibrosaSwinTransformer(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
        )
    elif mode == "valid":
        dataset = VoiceLibrosaSwinTransformer(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
        )
    elif mode == "test":
        dataset = VoiceLibrosaSwinTransformer(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
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

    dataset = VoiceCRNetData(
        "../../../datasets",
        "voice_data/voice_librosa/train_data",
        "annotation/annotation_training.pkl",
    )
    for i in range(len(dataset)):
        if i > 3:
            break
        a = dataset[i]
        print(a)
    # data_loader = DataLoader(dataset, batch_size=8, num_workers=0)
    # for i, batch in enumerate(data_loader):
    #     print(batch["aud_data"].shape, batch["aud_label"].shape)
    #     if i >= 20:
    #         break

