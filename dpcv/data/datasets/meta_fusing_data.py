import torch
import glob
import os
from torch.utils.data import DataLoader
from pathlib import Path
from .build import DATA_LOADER_REGISTRY
import os.path as opt
import numpy as np
import json


class MetaFuseingData:

    TRAITS_ID = {
        "O": 0, "C": 1, "E": 2, "A": 3, "N": 4,
    }

    def __init__(
        self, 
        data_root, 
        ann_dir,
        split, 
        session, 
        traits="OCEAN", 
    ):
        assert session in ["none", "talk", "animal", "lego", "ghost"], \
            "session should be in one of ['lego', 'talk', 'animal', 'ghost']"

        self.data_root = data_root
        self.ann_dir = ann_dir
        self.split = split
        self.session = session
        self.sample_ls = self.get_data_ls(split)
        self.traits2id = self.load_reverse_dict(split)
        self.id2meta = self.load_meta_dict(split)
        self.traits = [self.TRAITS_ID[t] for t in traits]
    
    def load_reverse_dict(self, split):
        file_path = os.path.join(self.ann_dir, f"parts_{split}_personality_vers.json")
        with open(file_path, 'r') as f:
            trait2id = json.load(f)
        return trait2id

    def load_meta_dict(self, split):
        file_path  = os.path.join(self.ann_dir, "metadata", f"metadata_{split}.json")
        with open(file_path, 'r') as f:
            id2meta = json.load(f)
        return id2meta

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        sample = torch.load(sample)
        label, feat = sample["video_label"], sample["video_frames_feat"]
        metadata_val = self.get_metadata_value(label)
        feat_meta = self.encode_feat_meta(feat, metadata_val)
        if not len(self.traits) == 5:
            label = sample["video_label"]
            sample["video_label"] = label[self.traits]
        sample["feat_meta"] = feat_meta
        return sample

    def encode_feat_meta(self, feat, metadata_val):
        metadata_val = metadata_val.repeat(1000, 1)
        ten = torch.cat([feat, metadata_val], dim=1)
        return ten

    def get_metadata_value(self, label):
        values_map = {
            "F": 0, "M": 1, "Primary": 1, "Secondary": 2, "Upper_secondary": 3,
            "Bachelor/uni_degree": 4, "Vocational/Training": 5,
            "Masters_degree" : 6, "Doctorate": 7
        }
        meta_data = self.get_metadata_dict(label)
        meta_values = [values_map[k] for k in [meta_data["gender"], meta_data["edu"]]]
        meta_values.append(int(meta_data["age"]))
        meta_values = torch.tensor(meta_values)
        return meta_values

    def get_metadata_dict(self, label):
        label_lst = label.numpy().tolist()
        label_key = "".join([str(l)[:3] for l in label_lst])
        id = self.traits2id[label_key]
        meta_data = self.id2meta[id]
        return meta_data


    def __len__(self):
        return len(self.sample_ls)

    def get_data_ls(self, split):
        data_dir = f"{split}/{self.session}_{split}"
        data_dir_path = Path(os.path.join(self.data_root, data_dir))
        data_ls_path = sorted(data_dir_path.rglob("*.pkl"))
        data_ls_sample = list(data_ls_path)
        return data_ls_sample


class MultiModelDL(MetaFuseingData):

    def encode_feat_meta(self, feat, metadata_val):
        feat = feat.squeeze()
        ten = torch.cat([feat, metadata_val], dim=0)
        return ten



@DATA_LOADER_REGISTRY.register()
def metadata_fuse_modal_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    if mode == "train":
        data_set = MetaFuseingData(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="train",
            session=cfg.DATA.SESSION,
        )
    elif mode == "valid":
        data_set = MetaFuseingData(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="valid",
            session=cfg.DATA.SESSION,
        )
    else:
        shuffle = False
        data_set = MetaFuseingData(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="test",
            session=cfg.DATA.SESSION,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def multi_model_metadata_fuse_data_loader(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    if mode == "train":
        data_set = MultiModelDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="train",
            session=cfg.DATA.SESSION,
        )
    elif mode == "valid":
        data_set = MultiModelDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="valid",
            session=cfg.DATA.SESSION,
        )
    else:
        shuffle = False
        data_set = MultiModelDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="test",
            session=cfg.DATA.SESSION,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def crnet_aud_metadata_fuse_dl(cfg, mode="train"):

    assert (mode in ["train", "valid", "trainval", "test", "full_test"]), \
        "'mode' should be 'train' , 'valid', 'trainval', 'test', 'full_test' "
    shuffle = cfg.DATA_LOADER.SHUFFLE

    if mode == "train":
        data_set = CRNMetaFeatDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="train",
            session=cfg.DATA.SESSION,
        )
    elif mode == "valid":
        data_set = CRNMetaFeatDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="valid",
            session=cfg.DATA.SESSION,
        )
    else:
        shuffle = False
        data_set = CRNMetaFeatDL(
            data_root=cfg.DATA.ROOT,
            ann_dir=cfg.DATA.ANN_DIR,
            split="test",
            session=cfg.DATA.SESSION,
        )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader




class CRNMetaFeatDL(MetaFuseingData):
    
    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        sample = torch.load(sample)
        label, feat = sample["video_label"], sample["video_frames_feat"]
        metadata_val = self.get_metadata_value(label)
        feat_meta = self.encode_feat_meta(feat, metadata_val)
        label = sample["video_label"]
        if not len(self.traits) == 5:
            sample["video_label"] = label[self.traits]
        if label.shape[0] != feat_meta.shape[0]:
            label = label[None].repeat(feat_meta.shape[0], 1)
        
        sample["feat_meta"] = feat_meta
        sample["video_label"] = label
        return sample

    def encode_feat_meta(self, feat, metadata_val):
        # print(feat.shape)
        if feat.shape[0] == 1:
            feat = feat.squeeze().mean(dim=0)
            # feat = torch.mean(feat, dim=0)
            ten = torch.cat([feat, metadata_val], dim=0)
        else:
            feat = feat.mean(dim=1)
            metadata = metadata_val[None].repeat(feat.shape[0], 1)
            ten = torch.cat([feat, metadata], dim=1)
        return ten

 



if __name__ == "__main__":
    # test setting
    import os; os.chdir("/root/DeepPersonality")

    data_set = MetaFuseingData(
        data_root="datasets/chalearn2021/model_output_features/02_hrnet_face",
        ann_dir="datasets/chalearn2021/annotation",
        split="test",
        session="animal",
    )
    for i in range(len(data_set)):
        print(data_set[i])


