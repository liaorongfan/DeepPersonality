import torch
from torch.utils.data import Dataset
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY
from torch.utils.data import DataLoader
import glob
import os
from tqdm import tqdm
import numpy as np
import pickle
import json
from math import ceil


class SoundFeatData(Dataset):
    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(self, data_path, mode, label_file, traits="OCEAN"):
        self.data_path = data_path
        self.mode = mode
        self.data_ls = self.get_data()
        self.annotation = self.parse_label(label_file)
        self.traits = [self.TRAITS_ID[t] for t in traits]

    def get_data(self):
        data_dir = os.path.join(self.data_path, f"{self.mode}_data")
        lst = glob.glob(f"{data_dir}/*.npy")
        return list(sorted(lst))

    def parse_label(self, label_file):

        with open(label_file, "rb") as f:
            annotation = pickle.load(f, encoding="latin1")
        return annotation

    def __getitem__(self, idx):
        data_path = self.data_ls[idx]
        data_arr = np.load(data_path)
        label = self.get_label(data_path)
        sample = {
            "feature": data_arr,
            "label": label,
        }
        return sample

    def get_label(self, data_path):
        video_path = data_path.replace(".wav.npy", "")
        video_name = f"{os.path.basename(video_path)}.mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return np.array(score)

    def __len__(self):
        return len(self.data_ls)


class TPSoundFeatData:

    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(self, data_path, mode, label_file, traits="OCEAN"):
        self.data_path = data_path
        self.mode = mode
        self.data_ls = self.get_data()
        self.session_id, self.parts_personality = self.parse_label(label_file)
        self.traits = [self.TRAITS_ID[t] for t in traits]

    def get_data(self):
        data_path = os.path.join(self.data_path, self.mode)
        data_lst = glob.glob(f"{data_path}/*/*/*.wav.npy")
        return list(sorted(data_lst))

    def parse_label(self, label_file):
        all_session_id, all_parts_personality = {}, {}
        ann_dir = os.path.join(self.data_path, "annotation", "all")
        for data_split in ["train", "valid", "test"]:
            session_id_path = os.path.join(ann_dir, f"sessions_{data_split}_id.json")
            with open(session_id_path, "r") as fo:
                session_id = json.load(fo)
            all_session_id.update(session_id)
            parts_personality = os.path.join(ann_dir, f"parts_{data_split}_personality.json")
            with open(parts_personality, "r") as fo:
                parts_personality = json.load(fo)
            all_parts_personality.update(parts_personality)
        return all_session_id, all_parts_personality

    def get_label(self, data_path):
        *_, session, part, sound = str(data_path).split("/")
        # if self.type == "face":
        sound = sound.replace(".wav.npy", "")[:3]
        sound = f"{sound}_T"
        # part = part.replace(self.task_mark, "T")
        participant_id = self.session_id[str(int(part))][sound]
        participant_trait = self.parts_personality[participant_id]
        score = [float(v) for v in participant_trait.values()]
        return np.array(score)

    def __getitem__(self, idx):
        data_path = self.data_ls[idx]
        data_arr = np.load(data_path)
        label = self.get_label(data_path)
        sample = {
            "feature": data_arr,
            "label": label,
        }
        return sample

    def __len__(self):
        return len(self.data_ls)


@DATA_LOADER_REGISTRY.register()
def sound_feat_dataloader(cfg, mode):
    assert mode in ["train", "valid", "test", "full_test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    SHUFFLE = True

    data_cfg = cfg.DATA
    sec_stage_cfg = cfg.DATA_LOADER.SECOND_STAGE
    if mode == "train":
        dataset = SoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.TRAIN_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
        )
    elif mode == "valid":
        dataset = SoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.VALID_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
        )
        SHUFFLE = False
    else:
        dataset = SoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.TEST_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
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


@DATA_LOADER_REGISTRY.register()
def sound_feat_tp_dataloader(cfg, mode):
    assert mode in ["train", "valid", "test", "full_test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    SHUFFLE = True

    data_cfg = cfg.DATA
    sec_stage_cfg = cfg.DATA_LOADER.SECOND_STAGE
    if mode == "train":
        dataset = TPSoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.TRAIN_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
        )
    elif mode == "valid":
        dataset = TPSoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.VALID_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
        )
        SHUFFLE = False
    else:
        dataset = TPSoundFeatData(
            data_path=data_cfg.ROOT,
            mode=mode,
            label_file=data_cfg.TEST_LABEL_DATA,  # method=sec_stage_cfg.METHOD,
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