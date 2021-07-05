from torch.utils.data import Dataset
import pickle
import os


class BimodalData(Dataset):
    """base class for bi-modal input data"""
    def __init__(self, data_root, img_dir, audio_dir, label_file):
        self.data_root = data_root
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.img_dir_ls = self.parse_img_dir(img_dir)  # every directory name indeed a video
        self.annotation = self.parse_annotation(label_file)

    def parse_img_dir(self, img_dir):
        img_dir_ls = os.listdir(os.path.join(self.data_root, img_dir))
        return img_dir_ls

    def parse_annotation(self, label_file):
        label_path = os.path.join(self.data_root, label_file)
        with open(label_path, "rb") as f:
            annotation = pickle.load(f, encoding="latin1")
        return annotation

    def _find_ocean_score(self, index):
        video_name = f"{self.img_dir_ls[index]}.mp4"
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

    def __len__(self):
        return len(self.img_dir_ls)

    def get_image_data(self, index):
        return self.img_dir_ls[index]

    def get_wave_data(self, index):
        img_dir_name = f"{self.img_dir_ls[index]}.wav"
        wav_path = os.path.join(self.data_root, self.audio_dir, img_dir_name)
        return wav_path
