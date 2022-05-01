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

    def __init__(self, data_root, data_split, task, data_type="frame", even_downsample=2000, trans=None, segment=False):
        self.data_root = data_root
        self.ann_dir = opt.join(data_root, "annotation", task)
        self.session_id, self.parts_personality = self.load_annotation(task, data_split)
        self.data_split = data_split
        self.task = task
        self.type = data_type
        self.data_dir = opt.join(data_root, data_split, f"{task}_{data_split}")
        self.sessions = os.listdir(self.data_dir)
        self.sample_size = even_downsample
        self.img_dir_ls = []
        self.task_mark = self.get_task_mark(task)
        if data_type == "frame":
            for dire in self.sessions:
                self.img_dir_ls.extend([f"{dire}/FC1_{self.task_mark}", f"{dire}/FC2_{self.task_mark}"])
        elif data_type == "face":
            for dire in self.sessions:
                self.img_dir_ls.extend([f"{dire}/FC1_{self.task_mark}_face", f"{dire}/FC2_{self.task_mark}_face"])
        else:
            raise TypeError(f"type should be 'face' or 'frame', but got {type}")

        if not segment:
            self.all_images = self.assemble_images()
        self.trans = trans

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_file = self.all_images[idx]
        img = Image.open(img_file)
        label = self.get_ocean_label(img_file)

        if self.trans:
            img = self.trans(img)

        return {"image": img, "label": torch.as_tensor(label, dtype=torch.float32)}

    def get_ocean_label(self, img_file):
        *_, session, part, frame = img_file.split("/")
        if self.type == "face":
            part = part.replace("_face", "")
        part = part.replace(self.task_mark, "T")
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

    def sample_img(self, img_dir):
        imgs = glob.glob(opt.join(self.data_dir, img_dir, "*.jpg"))
        if self.type == "frame":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[6:]))
        elif self.type == "face":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))

        # evenly sample to self.sample_size frames
        separate = np.linspace(0, len(imgs), self.sample_size, endpoint=False, dtype=np.int16)
        # index = random.choice(separate)
        selected_imgs = [imgs[idx] for idx in separate]
        # that will cost too much memory on disk
        # label = self.parse_label(selected_imgs[1])
        # labels = [label] * len(selected_imgs)
        return selected_imgs  # , labels

    def assemble_images(self):
        all_images = []
        for img_dir in self.img_dir_ls:
            all_images.extend(self.sample_img(img_dir))
        return all_images

    @staticmethod
    def get_task_mark(task):
        if task == "talk":
            return "T"
        elif task == "animal":
            return "A"
        elif task == "ghost":
            return "G"
        elif task == "lego":
            return "L"
        else:
            raise ValueError(
                f" task should be in one [talk, animal, ghost, lego]"
            )


@DATA_LOADER_REGISTRY.register()
def true_personality_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    transform = build_transform_spatial(cfg)
    dataset = Chalearn21FrameData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
        data_type=cfg.DATA.TYPE,
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
    train_dataset = Chalearn21FrameData(
        data_root="datasets/chalearn2021",
        data_split="train",
        task="talk",
    )
    print(len(train_dataset))
    print(train_dataset[1])

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

