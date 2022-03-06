import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os.path as opt
import glob
import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY


class Chalearn21FrameData(Dataset):

    def __init__(self, data_root, data_split, task, even_downsample=2000, trans=None):
        self.data_root = data_root
        self.ann_dir = opt.join(data_root, "annotation", task)
        self.session_id, self.parts_personality = self.load_annotation(task, data_split)
        self.data_split = data_split
        self.task = task
        self.data_dir = opt.join(data_root, data_split, f"{task}_{data_split}")
        self.sessions = os.listdir(self.data_dir)
        self.sample_size = even_downsample
        self.img_dirs = []
        for dire in self.sessions:
            self.img_dirs.extend([f"{dire}/FC1_T", f"{dire}/FC2_T"])
        self.trans = trans

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        img_file = self.sample_img(img_dir)
        img = Image.open(img_file)
        # img = np.array(Image.open(img_file))
        label = self.parse_label(img_file)

        if self.trans:
            img = self.trans(img)

        return {"image": img, "label": torch.as_tensor(label, dtype=torch.float32)}

    def parse_label(self, img_file):
        *_, session, part, frame = img_file.split("/")
        participant_id = self.session_id[str(int(session))][part]
        participant_trait = self.parts_personality[participant_id]
        participant_trait = np.array([float(v) for v in participant_trait.values()])
        return participant_trait

    def load_annotation(self, task, data_split):
        session_id_path = opt.join(self.ann_dir, f"sessions_{data_split}_id.json")
        with open(session_id_path, "r") as fo:
            session_id = json.load(fo)

        parts_personality = opt.join(self.ann_dir, f"parts_{data_split}_personality.json")
        with open(parts_personality, "r") as fo:
            parts_personality = json.load(fo)

        return session_id, parts_personality

    def sample_img(self, img_dir, even_downsample=2000):
        imgs = glob.glob(opt.join(self.data_dir, img_dir, "*.jpg"))
        imgs = sorted(imgs, key=lambda x: int(Path(x).stem[6:]))
        # evenly sample to self.sample_size frames
        separate = np.linspace(0, len(imgs), self.sample_size, endpoint=False, dtype=np.int16)
        index = random.choice(separate)
        return imgs[index]


@DATA_LOADER_REGISTRY.register()
def true_personality_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    transform = build_transform_spatial(cfg)
    dataset = Chalearn21FrameData(
        data_root="datasets/chalearn2021",
        data_split=mode,
        task="talk",
        trans=transform,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader



if __name__ == "__main__":
    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")
    # train_dataset = Chalearn21FrameData(
    #     data_root="datasets/chalearn2021",
    #     data_split="train",
    #     task="talk",
    # )
    # print(len(train_dataset))
    # print(train_dataset[1])

    test_dataset = Chalearn21FrameData(
        data_root="datasets/chalearn2021",
        data_split="test",
        task="talk",
    )
    print(len(test_dataset))
    print(test_dataset[1])

    val_dataset = Chalearn21FrameData(
        data_root="datasets/chalearn2021",
        data_split="val",
        task="talk",
    )
    print(len(val_dataset))
    print(val_dataset[1])

