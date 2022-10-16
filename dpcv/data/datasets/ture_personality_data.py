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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchaudio
from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.datasets.build import DATA_LOADER_REGISTRY


def norm(aud_ten):
    mean = aud_ten.mean()
    std = aud_ten.std()
    normed_ten = (aud_ten - mean) / (std + 1e-10)
    return normed_ten


class Chalearn21FrameData(Dataset):

    def __init__(
            self, data_root, data_split, task, data_type="frame",
            even_downsample=2000, trans=None, segment=False
    ):
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
        if (data_type == "frame") or (data_type == "audio"):
            for dire in self.sessions:
                self.img_dir_ls.extend([f"{dire}/FC1_{self.task_mark}", f"{dire}/FC2_{self.task_mark}"])
        elif data_type == "face":
            for dire in self.sessions:
                self.img_dir_ls.extend([f"{dire}/FC1_{self.task_mark}_face", f"{dire}/FC2_{self.task_mark}_face"])
        else:
            raise TypeError(f"type should be 'face' or 'frame' or 'audio', but got {type}")

        self.segment = segment
        if not data_type == "audio":
            self.all_images = self.assemble_images()
        self.trans = trans

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_file = self.all_images[idx]
        if not self.segment:
            img = Image.open(img_file)
            label = self.get_ocean_label(img_file)

            if self.trans:
                img = self.trans(img)
        else:
            img_ls = [Image.open(img) for img in img_file]
            label = self.get_ocean_label(img_file[0])
            if self.trans:
                img_ls = [self.trans(img) for img in img_ls]
            img = torch.stack(img_ls, dim=0)

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
            if not self.segment:
                all_images.extend(self.sample_img(img_dir))
            else:
                all_images.append(self.sample_img(img_dir))
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

    def get_file_path(self, idx):
        file = self.all_images[idx]
        if not self.segment:
            return file
        else:
            return file[0]


class Chlearn21AudioData(Chalearn21FrameData):
    def __init__(self, data_root, data_split, task, sample_len=244832, suffix_type="npy", data_type="audio"):
        super().__init__(data_root, data_split, task, data_type=data_type, segment=True)
        self.sample_len = sample_len
        self.suffix = suffix_type

    def __len__(self):
        return len(self.img_dir_ls)

    def __getitem__(self, idx):
        img_dir = self.img_dir_ls[idx]
        aud_data = self.sample_audio_data(img_dir)
        aud_label = self.get_ocean_label(img_dir)
        sample = {
            "aud_data": torch.as_tensor(aud_data, dtype=torch.float32),
            "aud_label": torch.as_tensor(aud_label, dtype=torch.float32),
        }
        return sample

    def sample_audio_data(self, img_dir):
        aud_file = opt.join(self.data_dir, f"{img_dir}.{self.suffix}")
        aud_data = np.load(aud_file)
        data_len = aud_data.shape[-1]
        start = np.random.randint(data_len - self.sample_len)
        end = start + self.sample_len
        return aud_data[:, :, start: end]

    def get_ocean_label(self, img_dir):
        *_, session, part = img_dir.split("/")
        part = part.replace(self.task_mark, "T")
        participant_id = self.session_id[str(int(session))][part]
        participant_trait = self.parts_personality[participant_id]
        participant_trait = np.array([float(v) for v in participant_trait.values()])
        return participant_trait


class Chalearn21AudioDataPath(Chlearn21AudioData):
    def __init__(self, data_root, data_split, task):
        super().__init__(data_root, data_split, task)

    def __getitem__(self, idx):
        img_dir = self.img_dir_ls[idx]
        aud_path = self.get_audio_path(img_dir)
        aud_label = self.get_ocean_label(img_dir)
        sample = {
            "aud_path": aud_path,
            "aud_label": torch.as_tensor(aud_label, dtype=torch.float32)
        }
        return sample

    def get_audio_path(self, img_dir):
        aud_path = os.path.join(self.data_dir, f"{img_dir}.wav")
        return aud_path


class Chalearn21LSTMAudioData(Chlearn21AudioData):

    def sample_audio_data(self, img_dir):
        aud_file = opt.join(self.data_dir, f"{img_dir}.wav_mfcc_mt.csv")
        wav_ft = np.loadtxt(aud_file, delimiter=",")
        return wav_ft


class Chalearn21InterpretAudioData(Chlearn21AudioData):

    def sample_audio_data(self, img_dir):
        aud_file = opt.join(self.data_dir, f"{img_dir}.wav")
        aud_data, sample_rate = torchaudio.load(aud_file)
        trans_aud = torchaudio.transforms.Resample(sample_rate, 4000)(aud_data[0, :].view(1, -1))
        trans_fft = torch.fft.fft(trans_aud)
        half_length = int(trans_aud.shape[-1] / 2)
        trans_fre = torch.abs(trans_fft)[..., :half_length]
        trans_fre = trans_fre[:, :40000]
        trans_fre_norm = norm(trans_fre)
        # if trans_fre_norm.shape[-1] < 30604:
        #     return self.get_wave_data(index - 1)
        return trans_fre_norm


class Chalearn21AudioVisualData(Chalearn21FrameData):
    sample_len = 50176

    def __getitem__(self, idx):
        img_file = self.all_images[idx]
        img = Image.open(img_file)
        label = self.get_ocean_label(img_file)
        wav = self.get_wave_data(img_file)
        if self.trans:
            img = self.trans(img)

        wav = torch.as_tensor(wav, dtype=img.dtype)
        label = torch.as_tensor(label, dtype=img.dtype)
        return {"image": img, "audio": wav, "label": label}

    def get_wave_data(self, img_file):
        dir_name = os.path.dirname(img_file)
        if self.type == "frame":
            aud_file = f"{dir_name}.npy"
        if self.type == "face":
            dir_name = dir_name.replace("_face", "")
            aud_file = f"{dir_name}.npy"
        aud_data = np.load(aud_file)
        data_len = aud_data.shape[-1]
        start = np.random.randint(data_len - self.sample_len)
        end = start + self.sample_len
        return aud_data[:, :, start: end]


class Chalearn21CRNetData(Chalearn21FrameData):
    sample_len = 244832

    def __getitem__(self, idx):
        img_file = self.all_images[idx]
        img = Image.open(img_file)

        loc_img = self.get_loc_img(img_file)

        label = self.get_ocean_label(img_file)
        label_cls_encode = self.cls_encode(label)

        wav = self.get_wave_data(img_file)
        if self.trans:
            img = self.trans["frame"](img)
            loc_img = self.trans["face"](loc_img)

        wav = torch.as_tensor(wav, dtype=img.dtype)
        label = torch.as_tensor(label, dtype=img.dtype)
        return {"glo_img": img, "loc_img": loc_img, "wav_aud": wav,
                "reg_label": label, "cls_label": label_cls_encode}

    def get_wave_data(self, img_file):
        dir_name = os.path.dirname(img_file)
        if self.type == "frame":
            aud_file = f"{dir_name}.npy"
        if self.type == "face":
            dir_name = dir_name.replace("_face", "")
            aud_file = f"{dir_name}.npy"
        aud_data = np.load(aud_file)
        data_len = aud_data.shape[-1]
        start = np.random.randint(data_len - self.sample_len)
        end = start + self.sample_len
        return aud_data[:, :, start: end]

    def get_loc_img(self, img_file):
        img_file = Path(img_file)
        img_id = img_file.stem.split("_")[-1]
        loc_img_dir = f"{img_file.parent}_face"
        loc_img_file = f"{loc_img_dir}/face_{img_id}.jpg"
        try:
            loc_img = Image.open(loc_img_file)
        except FileNotFoundError:
            loc_img_ls = list(Path(loc_img_dir).rglob("*.jpg"))
            loc_img_file = random.choice(loc_img_ls)
            loc_img = Image.open(loc_img_file)
        return loc_img

    @staticmethod
    def cls_encode(score):
        index = []
        for v in score:
            if v < -1:
                index.append(0)
            elif -1 <= v < 0:
                index.append(1)
            elif 0 <= v < 1:
                index.append(2)
            else:
                index.append(3)
        one_hot_cls = np.eye(4)[index]
        return one_hot_cls


class CRNetAudioTruePersonality(Chlearn21AudioData):

    def __init__(self, data_root, data_split, task, sample_len=244832):
        super().__init__(data_root, data_split, task, sample_len)

    def __getitem__(self, idx):
        img_dir = self.img_dir_ls[idx]
        aud_data = self.sample_audio_data(img_dir)
        aud_label = self.get_ocean_label(img_dir)
        label_cls = self.cls_encode(aud_label)
        return {
            "aud_data": torch.as_tensor(aud_data, dtype=torch.float32),
            "aud_label": torch.as_tensor(aud_label, dtype=torch.float32),
            "aud_label_cls": torch.as_tensor(label_cls, dtype=torch.float32),
        }

    @staticmethod
    def cls_encode(score):
        index = []
        for v in score:
            if v < -1:
                index.append(0)
            elif -1 <= v < 0:
                index.append(1)
            elif 0 <= v < 1:
                index.append(2)
            else:
                index.append(3)
        one_hot_cls = np.eye(4)[index]
        return one_hot_cls


class Chalearn21PersemonData(Chalearn21FrameData):

    def __init__(
            self, data_root, data_split, task, data_type, trans,
            emo_data_root, emo_img_dir, emo_label, emo_trans, segment=False,
    ):
        super().__init__(
            data_root, data_split, task, data_type=data_type,
            even_downsample=2000, trans=trans, segment=segment
        )
        self.emo_data_root = emo_data_root
        self.emo_img_dir = emo_img_dir
        self.emo_label = emo_label
        self.emo_trans = emo_trans
        self.emo_data_ls = self.emo_data_parser()

    def emo_data_parser(self):
        emo_label_path = os.path.join(self.emo_data_root, self.emo_label)
        video_files = [file for file in os.listdir(emo_label_path) if len(file) < 12]
        video_files_pt = [os.path.join(emo_label_path, video_file) for video_file in video_files]
        return video_files_pt

    def __getitem__(self, index):
        per_img_ls, per_lab_ls = self.gather_personality_data(index)
        emo_img_ls, emo_lab_ls = self.gather_emotion_data()
        if self.trans:
            per_img_ls = [self.trans(per_img) for per_img in per_img_ls]
        if self.emo_trans:
            emo_img_ls = [self.emo_trans(emo_img) for emo_img in emo_img_ls]

        per_imgs_ts = torch.stack(per_img_ls, 0)
        per_labs = torch.as_tensor(per_lab_ls, dtype=torch.float32)
        emo_imgs_ts = torch.stack(emo_img_ls, 0)
        emo_labs = torch.as_tensor(emo_lab_ls)
        sample = {
            "per_img": per_imgs_ts,
            "emo_img": emo_imgs_ts,
            "per_label": per_labs,
            "emo_label": emo_labs,
        }
        return sample

    def gather_emotion_data(self):
        file = random.choice(self.emo_data_ls)
        file_name = Path(file).stem
        img_dir = os.path.join(self.emo_data_root, self.emo_img_dir, file_name)
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        imgs = imgs[:100]
        imgs_pt = [os.path.join(img_dir, img) for img in imgs]
        with open(file, 'r') as f:
            frame_label = [map(lambda x: float(x), line.strip().split(",")) for line in f.readlines()[1:]]
        try:
            imgs_label = [list(frame_label[int(img_name.split(".")[0])]) for img_name in imgs]
        except:
            return self.gather_emotion_data()
        imgs_rgb = [Image.open(img_pt) for img_pt in imgs_pt]
        return imgs_rgb, imgs_label

    def gather_personality_data(self, index):
        img_files = self.all_images[index * 100: (index + 1) * 100]
        img_ls = []
        label_ls = []
        for img_file in img_files:
            img_ls.append(Image.open(img_file))
            label_ls.append(self.get_ocean_label(img_file))
        return img_ls, label_ls

    def __len__(self):
        return int(len(self.all_images) / 100)


class Chalearn21LSTMData(Chalearn21FrameData):
    def __init__(
            self, data_root, data_split, task, data_type="frame",
            even_downsample=2000, trans=None, segment=True,
    ):
        super().__init__(data_root, data_split, task, data_type, even_downsample, trans, segment)

    def __getitem__(self, idx):
        imgs_array_ls, file_ls = self._get_statistic_img_sample(idx)
        wav_ft = self._get_wav_sample(file_ls[0])
        anno_score = self.get_ocean_label(file_ls[0])
        if self.trans:
            imgs_ten_ls = []
            for img_arr in imgs_array_ls:
                img_ten = self.trans(img_arr)
                imgs_ten_ls.append(img_ten)
            imgs_ten = torch.stack(imgs_ten_ls, dim=0)
        else:
            imgs_ten = torch.as_tensor(imgs_array_ls)

        wav_ft = torch.as_tensor(wav_ft, dtype=imgs_ten.dtype)
        anno_score = torch.as_tensor(anno_score, dtype=imgs_ten.dtype)
        sample = {"image": imgs_ten, "audio": wav_ft, "label": anno_score}
        return sample

    def __len__(self):
        return len(self.img_dir_ls)

    def _get_statistic_img_sample(self, index):
        img_dir = self.img_dir_ls[index]
        imgs = glob.glob(opt.join(self.data_dir, img_dir, "*.jpg"))
        if self.type == "frame":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[6:]))
        elif self.type == "face":
            imgs = sorted(imgs, key=lambda x: int(Path(x).stem[5:]))
        if len(imgs) > 10:
            separate = np.linspace(0, len(imgs) - 1, 7, endpoint=True, dtype=np.int32)
            selected = [random.randint(separate[idx], separate[idx + 1]) for idx in range(6)]
            img_array_ls = []
            img_file_ls = []
            for idx in selected:
                img_pt = imgs[idx]
                img_array = Image.open(img_pt).convert("RGB")
                img_array_ls.append(img_array)
                img_file_ls.append(img_pt)
            return img_array_ls, img_file_ls
        else:
            raise ValueError("encountered bad input {}".format(self.img_dir_ls[index]))

    def _get_wav_sample(self, img_file):
        img_dir_name = os.path.dirname(img_file)
        if "face" in img_dir_name:
            img_dir_name = img_dir_name.replace("_face", "")
        wav_path = f"{img_dir_name}.wav_mfcc_mt.csv"
        wav_ft = np.loadtxt(wav_path, delimiter=",")
        return wav_ft


class Chalearn21LSTMVisualData(Chalearn21LSTMData):

    def __getitem__(self, idx):
        imgs_array_ls, file_ls = self._get_statistic_img_sample(idx)
        anno_score = self.get_ocean_label(file_ls[0])
        if self.trans:
            imgs_ten_ls = []
            for img_arr in imgs_array_ls:
                img_ten = self.trans(img_arr)
                imgs_ten_ls.append(img_ten)
            imgs_ten = torch.stack(imgs_ten_ls, dim=0)
        else:
            imgs_ten = torch.as_tensor(imgs_array_ls)

        anno_score = torch.as_tensor(anno_score, dtype=imgs_ten.dtype)
        sample = {"image": imgs_ten, "label": anno_score}
        return sample


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


@DATA_LOADER_REGISTRY.register()
def true_personality_audio_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    dataset = Chlearn21AudioData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_audio_bimodal_lstm_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    dataset = Chalearn21LSTMAudioData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_interpret_aud_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    dataset = Chalearn21InterpretAudioData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_crnet_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True
    transforms = build_transform_spatial(cfg)
    num_worker = cfg.DATA_LOADER.NUM_WORKERS if mode in ["valid", "train"] else 1
    dataset = Chalearn21CRNetData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
        trans=transforms,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_crnet_audio_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True
    num_worker = cfg.DATA_LOADER.NUM_WORKERS if mode in ["valid", "train"] else 1
    dataset = CRNetAudioTruePersonality(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_persemon_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True

    transforms = build_transform_spatial(cfg)
    persemon_dataset = Chalearn21PersemonData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,
        data_type=cfg.DATA.TYPE,  # "frame",
        trans=transforms,
        emo_data_root=cfg.DATA.VA_ROOT,  # "datasets",
        emo_img_dir=cfg.DATA.VA_DATA,  # "va_data/cropped_aligned",
        emo_label=cfg.DATA.VA_TRAIN_LABEL if mode == "train" else cfg.DATA.VA_VALID_LABEL,
        emo_trans=transforms,
    )
    data_loader = DataLoader(
        dataset=persemon_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_audio_visual_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True
    transforms = build_transform_spatial(cfg)
    num_worker = cfg.DATA_LOADER.NUM_WORKERS if mode in ["valid", "train"] else 1
    dataset = Chalearn21AudioVisualData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
        data_type=cfg.DATA.TYPE,
        trans=transforms
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_lstm_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True
    transforms = build_transform_spatial(cfg)
    num_worker = cfg.DATA_LOADER.NUM_WORKERS if mode in ["valid", "train"] else 0
    dataset = Chalearn21LSTMData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
        data_type=cfg.DATA.TYPE,
        trans=transforms
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def true_personality_lstm_visual_dataloader(cfg, mode):
    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "

    shuffle = False if mode in ["valid", "test", "full_test"] else True
    transforms = build_transform_spatial(cfg)
    num_worker = cfg.DATA_LOADER.NUM_WORKERS if mode in ["valid", "train"] else 0
    dataset = Chalearn21LSTMVisualData(
        data_root=cfg.DATA.ROOT,  # "datasets/chalearn2021",
        data_split=mode,
        task=cfg.DATA.SESSION,  # "talk"
        data_type=cfg.DATA.TYPE,
        trans=transforms
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_worker,
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

    # ==========================================================
    # test_dataset = Chalearn21FrameData(
    #     data_root="datasets/chalearn2021",
    #     data_split="test",
    #     task="talk",
    # )
    # print(len(test_dataset))
    # print(test_dataset[1])

    # ===========================================================
    # val_dataset = Chalearn21FrameData(
    #     data_root="datasets/chalearn2021",
    #     data_split="val",
    #     task="talk",
    # )
    # print(len(val_dataset))
    # print(val_dataset[1])

    # ===========================================================================================================
    # train_dataset = Chlearn21AudioData(
    #     data_root="datasets/chalearn2021",
    #     data_split="train",
    #     task="talk",
    # )
    #
    # print(len(train_dataset))
    # print(train_dataset[1])

    # ============================================================================================================
    # def face_image_transform():
    #     import torchvision.transforms as transforms
    #     norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    #     norm_std = [0.229, 0.224, 0.225]
    #     transforms = transforms.Compose([
    #         transforms.Resize(112),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(norm_mean, norm_std)
    #     ])
    #     return transforms
    #
    # transforms = face_image_transform()
    # persemon_dataset = Chalearn21PersemonData(
    #     data_root="datasets/chalearn2021", data_split="train", task="talk", data_type="frame", trans=transforms,
    #     emo_data_root="datasets", emo_img_dir="va_data/cropped_aligned",
    #     emo_label="va_data/va_label/VA_Set/Train_Set", emo_trans=transforms,
    # )
    # print(persemon_dataset[2])

    train_dataset = Chalearn21CRNetData(
        data_root="datasets/chalearn2021",
        data_split="train",
        task="talk",
    )

    print(len(train_dataset))
    print(train_dataset[1])
