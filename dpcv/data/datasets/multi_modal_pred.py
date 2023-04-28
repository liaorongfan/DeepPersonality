import torch
import glob
import os
from torch.utils.data import DataLoader
from pathlib import Path
from .build import DATA_LOADER_REGISTRY
import os.path as opt


class MultiModalData:

    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(self, data_root, split, mode, session, spectrum_channel=15, traits="OCEAN", 
        visual_clip=-1.0, audio_clip=-1.0, num_videos=-1,
    ):
        assert session in ["none", "talk", "animal", "lego", "ghost"], \
            "session should be in one of ['none', 'talk', 'animal', 'ghost'] or 'none'"
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.session = session
        self.spectrum_channel = spectrum_channel
        self.visual_clip = visual_clip
        self.audio_clip = audio_clip
        self.sample_ls = self.get_data_ls(split, mode)
        if num_videos > 0:
            self.sample_ls = self.sample_ls[: num_videos]
        self.traits = [self.TRAITS_ID[t] for t in traits]

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
            label = sample["label"]
            if not len(self.traits) == 5:
                sample["label"] = label[self.traits]

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
        if self.visual_clip > 0:
            data_ls_sample = []
            video_ls = list(data_dir_path.glob("*"))
            for video in video_ls:
                video_clip = list(video.glob("*.pkl"))[: self.visual_clip]
                data_ls_sample.extend(video_clip)
        else:
            data_ls_path = sorted(data_dir_path.rglob("*.pkl"))
            data_ls_sample = list(data_ls_path)
        # for sample in data_ls_path:
        #     data_ls_sample.append(sample)

        return data_ls_sample


class FoldMultiModalData:

    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(
        self, 
        source_data_root, ext_data_root, 
        split, mode, session, 
        spectrum_channel=15,
        traits="OCEAN",

    ):
        self.source_data_root = source_data_root
        self.ext_data_root = ext_data_root
        self.split = split
        self.mode = mode
        self.session = session
        self.spectrum_channel = spectrum_channel
        self.all_feat_data_dt = self.get_all_sub_video_feat_dict()
        self.data_dir = opt.join(source_data_root, split, f"{session}_{split}")
        self.fold_sub_video_ls = self.get_fold_data()
        # self.sub_video_features = []
        # if not mode == "audio":
        self.sub_video_features = self.assemble_video_features()
        self.traits = [self.TRAITS_ID[t] for t in traits]
        # print()
    
    def get_fold_data(self):

        fold_sub_video_ls = []
        video_ls = glob.glob(f"{self.data_dir}/*")
        task_mark = self.get_task_mark(self.session)
        for video in video_ls:
            fold_sub_video_ls.extend([
                f"{video}/FC1_{task_mark}", 
                f"{video}/FC2_{task_mark}",
            ])
        return fold_sub_video_ls

    def assemble_video_features(self):
        all_features = []
        for video in self.fold_sub_video_ls:
            k = "_".join(video.split("/")[-2:])
            feature_data = self.all_feat_data_dt[k]
            if self.mode == "audio":
                all_features.append(feature_data)
            else:
                frame_features = glob.glob(f"{feature_data}/*.pkl")
                all_features.extend(list(sorted(frame_features)))
        return all_features

    def get_task_mark(self, session):
        assert session in ["animal", "talk", "ghost", "lego"]
        return session[0].upper()
    
    def get_all_sub_video_feat_dict(self):
        data_dir_ls = []
        for split in ["train", "valid", "test"]:
            data_dir = f"{self.session}/{split}_{self.mode}"
            data_dir_ls.append(
                opt.join(self.ext_data_root, data_dir)
            )
        all_feature_data = []
        for path in data_dir_ls:
            all_feature_data.extend(
                # list(sorted(Path(path).rglob("*.pkl")))
                list(glob.glob(f"{path}/*"))
            )

        if self.mode == "audio":
            feature_data_dt = {
                opt.basename(feat).replace(".pkl", ""): feat
                for feat in all_feature_data
            }
        else:
            feature_data_dt = {
                opt.basename(feat).replace("_face", ""): feat
                for feat in all_feature_data
            }
        return feature_data_dt

    def __len__(self):

        return len(self.sub_video_features)

    def __getitem__(self, idx):
        # sub_video = self.fold_sub_video_ls[idx]
        # sample = self.get_sample_feature(sub_video)
        sample = self.sub_video_features[idx]

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
            label = sample["label"]
            if not len(self.traits) == 5:
                sample["label"] = label[self.traits]

        # data, label = sample["data"], sample["label"]
        return sample


class FoldMultiModalImpressionData:
    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(
            self,
            source_data_root, ext_data_root,
            split, mode,
            spectrum_channel=15,
            traits="OCEAN",

    ):
        self.source_data_root = source_data_root
        self.ext_data_root = ext_data_root
        self.split = split
        self.mode = mode
        self.spectrum_channel = spectrum_channel
        self.all_feat_data_dt = self.get_all_video_feat_dict()
        self.data_dir = self.get_data_dir()
        self.fold_video_ls = list(sorted(glob.glob(f"{self.data_dir}/*")))
        # self.sub_video_features = []
        # if not mode == "audio":
        self.traits = [self.TRAITS_ID[t] for t in traits]
        # print()

    def get_data_dir(self):
        if self.mode == "face":
            data_dir = opt.join(self.source_data_root, "image_data", f"{self.split}_data_face")
        elif self.mode == "frame":
            data_dir = opt.join(self.source_data_root, "image_data", f"{self.split}_data")
        else:
            data_dir = opt.join(self.source_data_root, "voice_data", "voice_librosa", f"{self.split}_data")
        return data_dir

    def get_all_video_feat_dict(self):
        data_dir_ls = []
        for split in ["train", "valid", "test"]:
            if self.mode == "face":
                data_dir = f"{split}_face"
            elif self.mode == "frame":
                data_dir = f"{split}_frame"
            else:
                data_dir = f"{split}_audio"
            data_dir_ls.append(
                opt.join(self.ext_data_root, data_dir)
            )
        all_feature_data = []
        for path in data_dir_ls:
            all_feature_data.extend(
                # list(sorted(Path(path).rglob("*.pkl")))
                list(glob.glob(f"{path}/*"))
            )

        feature_data_dt = {
            opt.basename(feat).replace(".pkl", ""): feat
            for feat in all_feature_data
        }
        return feature_data_dt

    def __len__(self):

        return len(self.fold_video_ls)

    def __getitem__(self, idx):
        # sub_video = self.fold_sub_video_ls[idx]
        # sample = self.get_sample_feature(sub_video)
        sample = self.fold_video_ls[idx]
        sample = self.get_feat_sample(sample)
        sample = torch.load(sample)
        if self.mode == "audio":
            feature = sample["feature"]
            sample_len = len(feature)
            if sample_len < 15:
                temp = torch.zeros(15, 128, dtype=feature.dtype)
                temp[: sample_len, :] = feature
                sample["feature"] = temp
            label = sample["label"]
            if not len(self.traits) == 5:
                sample["label"] = label[self.traits]

        # data, label = sample["data"], sample["label"]
        return sample

    def get_feat_sample(self, sample):
        name = os.path.basename(sample)
        if self.mode == "audio":
            name = name.replace(".npy", "").replace(".wav", "")
        feat_sample = self.all_feat_data_dt[name]
        return feat_sample


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
            visual_clip=cfg.DATA.VISUAL_CLIP,
            audio_clip=cfg.DATA.AUDIO_CLIP,
            num_videos=cfg.DATA.TRAIN_NUM_VIDEOS,
        )
    elif mode == "valid":
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="valid",
            mode=cfg.DATA.TYPE,
            session=cfg.DATA.SESSION,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
            visual_clip=cfg.DATA.VISUAL_CLIP,
            audio_clip=cfg.DATA.AUDIO_CLIP,
            num_videos=cfg.DATA.VALID_NUM_VIDEOS,
        )
    else:
        shuffle = False
        data_set = MultiModalData(
            data_root=cfg.DATA.ROOT,
            split="test",
            mode=cfg.DATA.TYPE,
            session=cfg.DATA.SESSION,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
            visual_clip=cfg.DATA.VISUAL_CLIP,
            audio_clip=cfg.DATA.AUDIO_CLIP,
            num_videos=cfg.DATA.TEST_NUM_VIDEOS,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def all_multi_modal_data_loader(cfg, mode="train"):
    from torch.utils.data.dataset import ConcatDataset

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE


    datasets = []
    # for session in ["ghost"]:
    for session in ["ghost", "animal", "talk", "lego"]:
        if mode == "train":
            data_set = MultiModalData(
                data_root=cfg.DATA.ROOT,
                split="train",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                traits=cfg.DATA.TRAITS
            )
        elif mode == "valid":
            data_set = MultiModalData(
                data_root=cfg.DATA.ROOT,
                split="valid",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                traits=cfg.DATA.TRAITS
            )
        else:
            shuffle = False
            data_set = MultiModalData(
                data_root=cfg.DATA.ROOT,
                split="test",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                traits=cfg.DATA.TRAITS
            )
        datasets.append(data_set)
    concat_dataset = ConcatDataset(datasets)
    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def all_fold_multi_modal_data_loader(cfg, mode="train"):
    from torch.utils.data.dataset import ConcatDataset

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    datasets = []
    # for session in ["ghost"]:
    for session in ["ghost", "animal", "talk", "lego"]:
        if mode == "train":
            data_set = FoldMultiModalData(
                source_data_root=cfg.DATA.ROOT,
                ext_data_root=cfg.DATA.FEATURE_ROOT,
                split="train",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                # traits=cfg.DATA.TRAITS
            )
        elif mode == "valid":
            data_set = FoldMultiModalData(
                source_data_root=cfg.DATA.ROOT,
                ext_data_root=cfg.DATA.FEATURE_ROOT,
                split="valid",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                # traits=cfg.DATA.TRAITS
            )
        else:
            shuffle = False
            data_set = FoldMultiModalData(
                source_data_root=cfg.DATA.ROOT,
                ext_data_root=cfg.DATA.FEATURE_ROOT,
                split="test",
                mode=cfg.DATA.TYPE,
                session=session,
                spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
                # traits=cfg.DATA.TRAITS
            )
        datasets.append(data_set)
    concat_dataset = ConcatDataset(datasets)
    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def fold_multi_modal_impression_data_loader(cfg, mode="train"):
    from torch.utils.data.dataset import ConcatDataset

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    datasets = []
    # for session in ["ghost"]:
    if mode == "train":
        data_set = FoldMultiModalImpressionData(
            source_data_root=cfg.DATA.ROOT,
            ext_data_root=cfg.DATA.FEATURE_ROOT,
            split="train",
            mode=cfg.DATA.TYPE,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
            # traits=cfg.DATA.TRAITS
        )
    elif mode == "valid":
        data_set = FoldMultiModalImpressionData(
            source_data_root=cfg.DATA.ROOT,
            ext_data_root=cfg.DATA.FEATURE_ROOT,
            split="valid",
            mode=cfg.DATA.TYPE,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
            # traits=cfg.DATA.TRAITS
        )
    else:
        shuffle = False
        data_set = FoldMultiModalImpressionData(
            source_data_root=cfg.DATA.ROOT,
            ext_data_root=cfg.DATA.FEATURE_ROOT,
            split="test",
            mode=cfg.DATA.TYPE,
            spectrum_channel=cfg.MODEL.SPECTRUM_CHANNEL,
            # traits=cfg.DATA.TRAITS
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


