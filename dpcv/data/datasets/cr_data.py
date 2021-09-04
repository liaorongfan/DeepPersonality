import torch
import os
import pickle
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
from PIL import Image
import random
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.datasets.transforms import set_crnet_transform


class CRNetData(VideoData):
    def __init__(self, data_root, img_dir, face_img_dir, audio_dir, label_file, transform=None):
        super().__init__(data_root, img_dir, audio_dir, label_file)
        self.transform = transform
        self.face_img_dir_ls = self.get_face_img_dir(face_img_dir)

    def get_face_img_dir(self, face_img_dir):
        dir_ls = os.listdir(os.path.join(self.data_root, face_img_dir))
        return dir_ls

    def __getitem__(self, idx):
        glo_img, loc_img, idx = self.get_imgs(idx)  # in case the idx changed
        wav_aud = self.get_wav_aud(idx)
        anno_score = self._find_ocean_score(idx)
        anno_cls_encode = self._cls_encode(anno_score)

        if self.transform:
            glo_img = self.transform(glo_img)
            loc_img = self.transform(loc_img)
        wav_aud = torch.as_tensor(wav_aud, dtype=glo_img.dtype)
        anno_score = torch.as_tensor(anno_score, dtype=glo_img.dtype)
        anno_cls_encode = torch.as_tensor(anno_cls_encode)

        sample = {
            "glo_img": glo_img, "loc_img": loc_img, "wav_aud": wav_aud,
            "reg_label": anno_score * 100, "cls_label": anno_cls_encode
        }
        return sample

    @staticmethod
    def _cls_encode(score):
        index = []
        for v in score:
            if 0 < v < 0.5:
                index.append(0)
            elif 0.5 <= v < 0.6:
                index.append(1)
            elif 0.6 <= v < 0.7:
                index.append(2)
            else:
                index.append(3)
        one_hot_cls = np.eye(4)[index]
        return one_hot_cls

    def get_imgs(self, idx):

        glo_img_dir = self.img_dir_ls[idx]
        loc_img_dir = (glo_img_dir + "_aligned").replace("data_01", "data_01_face")
        if os.path.basename(loc_img_dir) not in self.face_img_dir_ls:
            return self.get_imgs(idx + 1)
        loc_imgs = glob.glob(loc_img_dir + "/*.jpg")
        # according to the paper sample 32 frames per video
        separate = np.linspace(0, len(loc_imgs), 32, endpoint=False, dtype=np.int8)
        img_index = random.choice(separate)
        try:
            loc_img_pt = loc_imgs[img_index]
        except IndexError:
            loc_img_pt = loc_imgs[0]
        glo_img_pt = self._match_img(loc_img_pt)

        loc_img_arr = Image.open(loc_img_pt).convert("RGB")
        glo_img_arr = Image.open(glo_img_pt).convert("RGB")

        return glo_img_arr, loc_img_arr, idx

    @staticmethod
    def _match_img(loc_img_pt):
        img_dir = os.path.dirname(loc_img_pt).replace("_face", "").replace("_aligned", "")
        img_name, _ = os.path.basename(loc_img_pt).split(".")
        img_id = int(img_name[-3:])
        glo_img_name = "frame" + str(img_id) + ".jpg"
        return os.path.join(img_dir, glo_img_name)

    def get_wav_aud(self, index):
        img_dir_name = os.path.basename(self.img_dir_ls[index])
        audio_name = f"{img_dir_name}.wav.npy"
        wav_path = os.path.join(self.data_root, self.audio_dir, audio_name)
        wav_ft = np.load(wav_path, allow_pickle=True)
        return wav_ft


def make_data_loader(cfg, mode=None):
    assert (mode in ["train", "valid"]), " 'mode' only supports 'train' and 'valid'"
    transforms = set_crnet_transform()
    if mode == "train":
        dataset = CRNetData(
            "../datasets",
            "image_data/training_data_01",
            "image_data/training_data_01_face",
            "voice_data/training_data_244832",
            "annotation/annotation_training.pkl",
            transforms
        )
    else:
        dataset = CRNetData(
            "../datasets",
            "image_data/validation_data_01",
            "image_data/validation_data_01_face",
            "voice_data/validation_data_244832",
            "annotation/annotation_validation.pkl",
            transforms
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4  # cfg.NUM_WORKS
    )
    return data_loader


if __name__ == "__main__":
    # trans = set_crnet_transform()
    # data_set = CRNetData(
    #     "../../../datasets",
    #     "ImageData/trainingData",
    #     "VoiceData/trainingData_244832",
    #     "annotation_training.pkl",
    #     trans
    # )
    # # print(len(data_set))
    # # print(data_set[2])
    # for item in data_set[2].values():
    #     print(item.shape)
    # # print(data_set._statistic_img_sample(1))
    # # print(data_set._get_wav_sample(1))

    loader = make_data_loader("", mode="train")
    for i, sample in enumerate(loader):
        # if i > 5:
        #     break
        for item in sample.values():
            print(item.shape)
