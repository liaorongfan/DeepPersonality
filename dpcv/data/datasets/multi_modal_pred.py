import torch
import glob
from torch.utils.data import DataLoader
from .build import DATA_LOADER_REGISTRY


class MultiModalData:

    def __init__(self, data_root, split, mode):
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.sample_ls = self.get_data_ls(split, mode)

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        if self.mode == "audio":
            feature = sample["feature"]
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
        data_dir = f"{split}_{mode}"
        data_dir_path = os.path.join(self.data_root, data_dir)
        data_ls_path = glob.glob(f"{data_dir_path}/*.pkl")
        data_ls_sample = []
        for sample in data_ls_path:
            data_ls_sample.append(torch.load(sample))
        return data_ls_sample


@DATA_LOADER_REGISTRY.register()
def multi_modal_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    if mode == "train":
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="train",
            mode=cfg.DATA.TYPE,
        )
    elif mode == "valid":
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="valid",
            mode=cfg.DATA.TYPE,
        )
    else:
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="test",
            mode=cfg.DATA.TYPE,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
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


