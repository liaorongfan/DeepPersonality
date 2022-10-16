import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
from random import shuffle


class VideoData(Dataset):
    """base class for bi-modal input data"""
    def __init__(self, data_root, img_dir, label_file, audio_dir=None, parse_img_dir=True, parse_aud_dir=False):
        self.data_root = data_root
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.annotation = self.parse_annotation(label_file)
        if parse_img_dir:
            self.img_dir_ls = self.parse_data_dir(img_dir)  # every directory name indeed a video
        if parse_aud_dir:
            self.aud_file_ls = self.parse_data_dir(audio_dir)

    def parse_data_dir(self, data_dir):
        """

        Args:
            data_dir:(Str or List[Str, ]) training audio data directory or train and valid data directory

        Returns:
            img_dir_path:(List[Str, ]) a list contains the path of image files
        """
        if isinstance(data_dir, list):
            data_dir_path = []
            for dir_i in data_dir:
                data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, dir_i)))
                data_dir_path.extend([os.path.join(self.data_root, dir_i, item) for item in data_dir_ls])
        else:
            data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, data_dir)))
            data_dir_path = [os.path.join(self.data_root, data_dir, item) for item in data_dir_ls]
        return data_dir_path

    def parse_annotation(self, label_file):
        """
            args:(srt / list[str, str]) annotation file path
        """
        if isinstance(label_file, list):
            assert len(label_file) == 2, "only support join train and validation data"
            anno_list = []
            for label_i in label_file:
                label_path = os.path.join(self.data_root, label_i)
                with open(label_path, "rb") as f:
                    anno_list.append(pickle.load(f, encoding="latin1"))
            for key in anno_list[0].keys():
                anno_list[0][key].update(anno_list[1][key])
            annotation = anno_list[0]
        else:
            label_path = os.path.join(self.data_root, label_file)
            with open(label_path, "rb") as f:
                annotation = pickle.load(f, encoding="latin1")
        return annotation

    def get_ocean_label(self, index):
        video_path = self.img_dir_ls[index]
        video_name = f"{os.path.basename(video_path)}.mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return score

    def __getitem__(self, index):
        raise NotImplementedError

    def get_image_data(self, index):
        return self.img_dir_ls[index]

    def get_wave_data(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_dir_ls)
