import os
from bi_modal_data import VideoData
import torch
import torchaudio


def aud_transform():
    return torchaudio.transforms.Resample(44100, 4000)


def norm(aud_ten):
    mean = aud_ten.mean()
    std = aud_ten.std()
    normed_ten = (aud_ten - mean) / (std + 1e-10)
    return normed_ten


class InterpretabilityAudio(VideoData):
    def __init__(self, data_root, aud_dir, label_file):
        super().__init__(
            data_root, img_dir=None, audio_dir=aud_dir, label_file=label_file,
            parse_img_dir=False,
            parse_aud_dir=True,
        )

    def __getitem__(self, index):
        aud_data = self.get_wave_data(index)
        label = self.get_ocean_score(index)
        return aud_data, label

    def get_wave_data(self, index):
        aud_file = self.aud_file_ls[index]
        aud_data, sample_rate = torchaudio.load(aud_file)
        trans_aud = torchaudio.transforms.Resample(sample_rate, 4000)(aud_data[0, :].view(1, -1))
        trans_fft = torch.fft.fft(trans_aud)
        half_length = int(trans_aud.shape[-1] / 2)
        trans_fre = torch.abs(trans_fft)[..., half_length]
        trans_fre_norm = norm(trans_fre)
        return trans_fre_norm

    def get_ocean_score(self, index):
        aud_file = self.aud_dir_ls[index]
        video_name = f"{os.path.basename(aud_file)}.mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return torch.tensor(score)

    def __len__(self):
        return len(self.aud_file_ls)

