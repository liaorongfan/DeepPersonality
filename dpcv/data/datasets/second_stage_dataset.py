import torch
from torch.utils.data import Dataset
from .build import DATA_LOADER_REGISTRY
from torch.utils.data import DataLoader
import glob


class SpectrumData(Dataset):

    def __init__(self, data_path):
        try:
            self.sample_ls = torch.load(data_path)
        except FileNotFoundError:
            data_segments = sorted(glob.glob(data_path.replace(".pkl", "*.pkl")))
            self.sample_ls = []
            for seg in data_segments:
                self.sample_ls.extend(torch.load(seg))

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        amp_spectrum = torch.as_tensor(sample["amp_spectrum"], dtype=torch.float32)
        pha_spectrum = torch.as_tensor(sample["pha_spectrum"], dtype=torch.float32)
        spectrum = torch.stack([amp_spectrum, pha_spectrum], dim=0)
        sample = {"spectrum": spectrum, "label": sample["video_label"]}
        return sample

    def __len__(self):
        return len(self.sample_ls)


class StatisticData(Dataset):

    def __init__(self, data_path):
        self.sample_dict = torch.load(data_path)

    def __getitem__(self, idx):
        sample = {
            "statistic": self.sample_dict["video_statistic"][idx],
            "label": self.sample_dict["video_label"][idx],
        }
        return sample

    def __len__(self):
        return len(self.sample_dict["video_label"])


@DATA_LOADER_REGISTRY.register()
def statistic_data_loader(cfg, mode):
    assert mode in ["train", "valid", "test", "full_test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    SHUFFLE = True

    data_cfg = cfg.DATA
    if mode == "train":
        dataset = StatisticData(data_cfg.TRAIN_IMG_DATA)
    elif mode == "valid":
        dataset = StatisticData(data_cfg.VALID_IMG_DATA)
        SHUFFLE = False
    else:
        dataset = StatisticData(data_cfg.TEST_IMG_DATA)
        SHUFFLE = False
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        num_workers=loader_cfg.NUM_WORKERS,
        shuffle=SHUFFLE
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def spectrum_data_loader(cfg, mode):
    assert mode in ["train", "valid", "test", "full_test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    SHUFFLE = True  # when at test time don't shuffle will get a slightly better result

    data_cfg = cfg.DATA
    if mode == "train":
        dataset = SpectrumData(data_cfg.TRAIN_IMG_DATA)
    elif mode == "valid":
        dataset = SpectrumData(data_cfg.VALID_IMG_DATA)
        SHUFFLE = False
    else:
        dataset = SpectrumData(data_cfg.TEST_IMG_DATA)
        SHUFFLE = False
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        num_workers=loader_cfg.NUM_WORKERS,
        shuffle=SHUFFLE
    )
    return data_loader
