import torch
from torch.utils.data import Dataset
from .build import DATA_LOADER_REGISTRY
from torch.utils.data import DataLoader


class SpectrumData(Dataset):

    def __init__(self, data_path):
        self.sample_ls = torch.load(data_path)

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        return {
            "amp_spectrum": sample["amp_spectrum"],
            "pha_spectrum": sample["pha_spectrum"],
            "label": sample["video_label"],
        }

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

    data_cfg = cfg.DATA
    if mode == "train":
        dataset = StatisticData(data_cfg.TRAIN_IMG_DATA)
    elif mode == "valid":
        dataset = StatisticData(data_cfg.VALID_IMG_DATA)
    else:
        dataset = StatisticData(data_cfg.TEST_IMG_DATA)

    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        num_workers=loader_cfg.NUM_WORKERS,
        shuffle=loader_cfg.SHUFFLE
    )
    return data_loader


