import torch
import glob
import os
from torch.utils.data import DataLoader
from pathlib import Path
from .build import DATA_LOADER_REGISTRY


class MultiModalData:

    def __init__(self, data_root, split, mode, session, spectrum_channel=15):
        assert session in ["none", "talk", "animal", "lego", "ghost"], \
            "session should be in one of ['none', 'talk', 'animal', 'ghost'] or 'none'"
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.session = session
        self.spectrum_channel = spectrum_channel
        self.sample_ls = self.get_data_ls(split, mode)

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        sample = torch.load(sample)
        if self.mode == "audio":
            feature = sample["feature"]
            if self.session in ["talk", "animal", "lego", "ghost"]:
                temp = feature[:self.spectrum_channel]
                sample["feature"] = temp
            else:
                sample_len = len(feature)
                if sample_len < 15:
                    temp = torch.zeros(15, 128, dtype=feature.dtype)
                    temp[: sample_len, :] = feature
                    sample["feature"] = temp

        # data, label = sample["data"], sample["label"]
        return sample

    def __len__(self):
        return len(self.sample_ls)

    def get_data_ls(self, split, mode):
        if self.session in ["talk", "animal", "lego", "ghost"]:
            data_dir = f"{self.session}/{split}_{mode}"
        else:
            data_dir = f"{split}_{mode}"
        data_dir_path = Path(os.path.join(self.data_root, data_dir))
        data_ls_path = sorted(data_dir_path.rglob("*.pkl"))
        data_ls_sample = list(data_ls_path)
        # for sample in data_ls_path:
        #     data_ls_sample.append(sample)

        return data_ls_sample


@DATA_LOADER_REGISTRY.register()
def multi_modal_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    if mode == "train":
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="train",
            mode=cfg.DATA.TYPE,
            session=cfg.DATA.SESSION,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
        )
    elif mode == "valid":
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="valid",
            mode=cfg.DATA.TYPE,
            session=cfg.DATA.SESSION,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
        )
    else:
        shuffle = False
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="test",
            mode=cfg.DATA.TYPE,
            session=cfg.DATA.SESSION,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


if __name__ == "__main__":
    # test setting
    import os; os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")

    data_set = MultiModalData(
        data_root="datasets/extracted_feature_impression",
        mode="frame",
        split="train",
    )


