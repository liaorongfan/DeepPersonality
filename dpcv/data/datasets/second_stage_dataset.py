import torch
from torch.utils.data import Dataset
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
from torch.utils.data import DataLoader
import glob
import os
from tqdm import tqdm
import numpy as np
from math import ceil


class SecondStageData(Dataset):

    def __init__(self, data_dir, data_type="pred", method="statistic", used_frame=1000, top_n_sample=600, sample=False):
        assert data_type in ["pred", "feat"]
        assert method in ["statistic", "spectrum"]

        self.data_type = f"video_frames_{data_type}"
        self.process_method = method
        data_root, data_split = os.path.split(data_dir)
        # self.signal_num = 512
        self.save_to = os.path.join(data_root, f"{self.data_type}_{self.process_method}_{data_split}.pkl")
        self.used_frame = used_frame
        self.top_n_sample = top_n_sample
        self.sample = sample
        self.data_preprocess(data_dir)  # save processed data to disk first
        self.data_ls = self.load_data()

    def load_data(self):
        # save_to = os.path.join(data_root, f"{self.data_type}_{self.process_method}_{data_split}.pkl")
        data_ls = []
        # use cached data
        # print(f"using cached data {self.save_to}")
        try:
            data_ls = torch.load(self.save_to)
        except FileNotFoundError:
            data_segments = sorted(glob.glob(self.save_to.replace(".pkl", "*.pkl")))
            for seg in data_segments:
                data_ls.extend(torch.load(seg))
        return data_ls

    def data_preprocess(self, data_dir):
        if os.path.exists(self.save_to) or os.path.exists(self.save_to.replace(".pkl", "_1.pkl")):
            return

        print(
            f"preprocessing data [{data_dir}] \n"
            f"[{self.data_type}] by [{self.process_method}]"
        )
        data_ls = []
        sample_pth_ls = sorted(glob.glob(f"{data_dir}/*.pkl"))
        sample_num = len(sample_pth_ls)
        seg_id = 0
        for i, sample_pth in enumerate(tqdm(sample_pth_ls)):
            sample = torch.load(sample_pth)
            data, label = sample[self.data_type], sample["video_label"]  # data: (382, 2048) label: (5,)

            if self.process_method == "statistic":
                data = self.statistic_process(data)
            elif self.process_method == "spectrum":
                data, valid = self.spectrum_process(data)  # data: (382, 2048) label: (5,)
                if not valid:
                    print(f"{sample_pth} not valid with data shape {data.shape}")
                    continue
            else:
                raise NotImplementedError

            sample_train = {"id": i, "data": data, "label": label}
            data_ls.append(sample_train)
            # signal_num = data_ls[0]["data"].shape[1]
            # if signal_num >= 1024:
            # if self.signal_num != signal_num:
            #     self.signal_num = signal_num
            # for large feature save 1000 item every time in case of memory issue
            # last_seg = ceil(len(sample_pth_ls) / 1000)
            data_seg = 2000
            if len(data_ls) == data_seg:
                seg_id += 1
                torch.save(data_ls[:data_seg], self.save_to.replace(".pkl", f"_{seg_id}.pkl"))
                data_ls = data_ls[data_seg:]
            elif i == sample_num - 1:
                seg_id += 1
                torch.save(data_ls, self.save_to.replace(".pkl", f"_{seg_id}.pkl"))

        # if self.signal_num < 1024:
        #     torch.save(data_ls, self.save_to)

    def __getitem__(self, idx):
        training_sample = self.data_ls[idx]
        return training_sample

    @staticmethod
    def statistic_process(data):
        assert isinstance(data, torch.Tensor), "the input data should be torch.Tensor"
        max_0, _ = data.max(dim=0)
        min_0, _ = data.min(dim=0)
        mean_0 = data.mean(dim=0)
        std_0 = data.std(dim=0)

        data_first_order = data[1:, :] - data[:-1, :]
        max_1, _ = data_first_order.max(dim=0)
        min_1, _ = data_first_order.min(dim=0)
        mean_1 = data_first_order.mean(dim=0)
        std_1 = data_first_order.std(dim=0)

        data_sec_order = data[2:, :] - data[:-2, :]
        max_2, _ = data_sec_order.max(dim=0)
        min_2, _ = data_sec_order.min(dim=0)
        mean_2 = data_sec_order.mean(dim=0)
        std_2 = data_sec_order.std(dim=0)

        statistic_representation = torch.stack(
            [
                max_0, min_0, mean_0, std_0,
                max_1, min_1, mean_1, std_1,
                max_2, min_2, mean_2, std_2
            ],
            dim=0,
        )

        return statistic_representation

    def spectrum_process(self, data):   # data: (382, 2048)
        amp_spectrum, pha_spectrum = [], []
        traits_representation = data.T.cpu()  # traits_represen: (2048, 382)
        for trait_i in traits_representation:
            amp, pha, valid = self.select_pred_spectrum(trait_i.numpy()[None, :])  # trait_i (1, 382)
            amp_spectrum.append(amp)
            pha_spectrum.append(pha)
        spectrum_data = {
            "amp_spectrum": np.concatenate(amp_spectrum, axis=0),  # (2048, 80)
            "pha_spectrum": np.concatenate(pha_spectrum, axis=0),  # (2048, 80)
        }
        spectrum_data = np.stack(                                  # (2, 2048, 80)
            [spectrum_data["amp_spectrum"],
             spectrum_data["pha_spectrum"]],
            axis=0,
        )
        return spectrum_data, valid

    def select_pred_spectrum(self, data):
        # for one trait there n prediction from n frames
        # data: (1, n)  eg:（1， 382）
        valid = True
        if self.sample:
            indexes = np.linspace(0, data.shape[1], 100, endpoint=False, dtype=np.int16)
            data = data[:, indexes]
        else:
            data = data[:, :self.used_frame]
        pred_fft = np.fft.fft2(data)  # pred_fft (1, 382)  complex num
        length = int(len(pred_fft[0]) / 2)
        amp, pha = np.abs(pred_fft),  np.angle(pred_fft)  # amp:(1, 382) pha:(1, 382)
        # include symmetry point
        if self.top_n_sample < length:
            amp[:, self.top_n_sample - 1] = amp[:, length]
            pha[:, self.top_n_sample - 1] = pha[:, length]

        amp_feat = amp[:, :self.top_n_sample]  # amp_feat:(1: 80) , pha_feat:(1: 80)
        pha_feat = pha[:, :self.top_n_sample]
        if len(amp_feat[0]) != self.top_n_sample:
            valid = False

        return amp_feat.astype("float32"), pha_feat.astype("float32"), valid

    def __len__(self):
        return len(self.data_ls)


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


@DATA_LOADER_REGISTRY.register()
def second_stage_data(cfg, mode):
    assert mode in ["train", "valid", "test", "full_test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    SHUFFLE = True

    data_cfg = cfg.DATA
    sec_stage_cfg = cfg.DATA_LOADER.SECOND_STAGE
    if mode == "train":
        dataset = SecondStageData(
            data_dir=data_cfg.TRAIN_IMG_DATA,
            data_type=sec_stage_cfg.TYPE,
            method=sec_stage_cfg.METHOD,
        )
    elif mode == "valid":
        dataset = SecondStageData(
            data_dir=data_cfg.VALID_IMG_DATA,
            data_type=sec_stage_cfg.TYPE,
            method=sec_stage_cfg.METHOD,
        )
        SHUFFLE = False
    else:
        dataset = SecondStageData(
            data_dir=data_cfg.TEST_IMG_DATA,
            data_type=sec_stage_cfg.TYPE,
            method=sec_stage_cfg.METHOD,
        )
        SHUFFLE = False
    loader_cfg = cfg.DATA_LOADER
    data_loader = DataLoader(
        dataset,
        batch_size=loader_cfg.TRAIN_BATCH_SIZE,
        num_workers=loader_cfg.NUM_WORKERS,
        shuffle=SHUFFLE,
        drop_last=cfg.DATA_LOADER.DROP_LAST,
    )
    return data_loader


if __name__ == "__main__":
    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")

    dataset = SecondStageData(
        data_dir="datasets/second_stage/hrnet_extract/test",
        data_type="feat",
        method="spectrum",
    )
    # dataset[1]
