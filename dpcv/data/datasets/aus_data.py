import glob
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import os
import json
import numpy as np
from .build import DATA_LOADER_REGISTRY


class AUsDataset:

    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    AUs = [
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06',
        'AU07', 'AU09', 'AU10', 'AU12', 'AU14',
        'AU15', 'AU17', 'AU20', 'AU23', 'AU25',
        'AU26', 'AU45'
    ]

    def __init__(
        self, data_root, split, au, label_file, session="",
        min_frame_num=360, traits="OCEAN", spec_sample=120
    ):
        self.data_root = data_root
        self.split = split
        self.au = au
        self.min_frame_num = min_frame_num
        self.top_n_sample = spec_sample
        self.session = session
        if session:
            self.all_csv_files = self.get_tp_csv_files()
            self.session_id, self.parts_personality = self.parse_tp_label_file()
        else:
            self.all_csv_files = self.get_ip_csv_files()
            self.label_dict = self.parse_ip_label_file(label_file)
        self.au_data, self.au_label = self.get_data()
        self.traits = [self.TRAITS_ID[t] for t in traits]

    def __getitem__(self, idx):
        return self.au_data[idx], self.au_label[idx]
    
    def __len__(self):
        return len(self.au_data)

    def get_tp_csv_files(self):
        data_dir = os.path.join(self.data_root, "chalearn21", "aus")
        au_csv_ls = []
        dirs = [
            path for path in glob.glob(f"{data_dir}/*/*")
            if self.split in path
        ]
        for di in dirs:
            csv_ls = list(Path(di).rglob("*.csv"))
            au_csv_ls.extend(csv_ls)
        return au_csv_ls

    def get_ip_csv_files(self):
        data_dir = os.path.join(
            self.data_root, "image_data", f"{self.split}_data_face_aus"
        )
        au_csv_ls = sorted(glob.glob(f"{data_dir}/*/*.csv"))
        return au_csv_ls

    def get_data(self):

        data_lst, label_lst = [], []
        for csv in self.all_csv_files:
            au_data = self.get_au_data(csv)
            if len(au_data) < self.min_frame_num:
                continue

            au_feat, valid = self.select_spectrum(au_data[None])
            if not valid:
                continue
            au_label = self.get_au_label(csv)
            data_lst.append(au_feat)
            label_lst.append(au_label)

        return data_lst, label_lst

    def get_au_label(self, csv):
        if self.session:
            *_, session, part, frame = str(csv).split("/")
            # if self.type == "face":
            part = part.replace("_face", "")[:3]
            # part = part.replace(self.task_mark, "T")
            participant_id = self.session_id[str(int(session))][part]
            participant_trait = self.parts_personality[participant_id]
            score = [float(v) for v in participant_trait.values()]
        else:
            video_name = f"{Path(csv).stem}.mp4"
            score = [
                self.label_dict["openness"][video_name],
                self.label_dict["conscientiousness"][video_name],
                self.label_dict["extraversion"][video_name],
                self.label_dict["agreeableness"][video_name],
                self.label_dict["neuroticism"][video_name],
            ]
        return np.array(score)

    def get_au_data(self, csv):
        aus_dict = {}
        with open(csv, 'r') as f:
            lines = [line.strip().split(",") for line in f.readlines()]
            aus_names = [key.strip() for key in lines[0]]
            val = [list(map(float, line)) for line in lines[1:]]
            val = np.array(val)
            for idx, key in enumerate(aus_names):
                aus_dict[key] = val[:, idx]
        return aus_dict[f"{self.au}_r"]

    @staticmethod
    def parse_ip_label_file(label_file):
        with open(label_file, 'rb') as f:
            ann = pickle.load(f, encoding="latin1")
        return ann

    def parse_tp_label_file(self, task=None):

        all_session_id, all_parts_personality = {}, {}
        ann_dir = os.path.join(self.data_root, "chalearn21", "annotation", "all")
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

    def select_spectrum(self, data):
        # for one trait there n prediction from n frames
        # data: (1, n)  eg:（1， 382）
        valid = True
        pred_fft = np.fft.fft2(data)  # pred_fft (1, 382)  complex num
        length = int(len(pred_fft[0]) / 2)
        amp, pha = np.abs(pred_fft), np.angle(pred_fft)  # amp:(1, 382) pha:(1, 382)
        # include symmetry point
        if self.top_n_sample < length:
            amp[:, self.top_n_sample - 1] = amp[:, length]
            pha[:, self.top_n_sample - 1] = pha[:, length]

        amp_feat = amp[:, :self.top_n_sample]  # amp_feat:(1: 80) , pha_feat:(1: 80)
        pha_feat = pha[:, :self.top_n_sample]
        feat = (amp_feat + pha_feat) / 2
        if len(amp_feat[0]) != self.top_n_sample:
            valid = False

        feat = feat[0].astype("float32")

        return feat, valid


@DATA_LOADER_REGISTRY.register()
def au_dataloader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    if mode == "train":
        dataset = AUsDataset(
            data_root=cfg.DATA.ROOT,
            split="train",
            au=cfg.DATA.AU,
            label_file=cfg.DATA.TRAIN_LABEL_DATA,
            session=cfg.DATA.SESSION,
        )
    elif mode == "valid":
        dataset = AUsDataset(
            data_root=cfg.DATA.ROOT,
            split="valid",
            au=cfg.DATA.AU,
            label_file=cfg.DATA.VALID_LABEL_DATA,
            session=cfg.DATA.SESSION,
        )
    else:
        shuffle = False
        dataset = AUsDataset(
            data_root=cfg.DATA.ROOT,
            split="test",
            au=cfg.DATA.AU,
            label_file=cfg.DATA.TEST_LABEL_DATA,
            session=cfg.DATA.SESSION,
        )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


if __name__ == '__main__':
    dataset = AUsDataset(
        data_root="datasets", split="test", au="AU01", session="",
        label_file="datasets/annotation/annotation_test.pkl"
    )

