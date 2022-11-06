"""
Temporary tool to extract wave sample features using librosa
"""
import os
import librosa
import opensmile
import numpy as np
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from dpcv.data.datasets.bi_modal_data import VideoData


class WaveProcessor(VideoData):

    def __init__(self, mode, save_to, *args, **kwargs):
        super(WaveProcessor, self).__init__(*args, **kwargs)
        self.mode = mode
        self.saved_file = save_to
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        if mode == "opensmile":
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )

    def __getitem__(self, idx):
        wav_file_path = self.aud_file_ls[idx]
        video_name = os.path.basename(wav_file_path)  # .replace(".wav", "")
        if self.mode == "librosa":
            self.librosa_extract(wav_file_path, video_name)
        elif self.mode == "logfbank":
            self.logfbank_extract(wav_file_path, video_name)
        elif self.mode == "opensmile":
            self.opensmile_extract(wav_file_path, video_name)

    def __len__(self):
        return len(self.aud_file_ls)

    def librosa_extract(self, wav_file_path, video_name):
        try:
            # sample rate 16000 Hz
            wav_ft = librosa.load(wav_file_path, 16000)[0][None, None, :]  # output_shape = (1, 1, 244832)
            # wav_ft = librosa.load(wav_path, 3279)[0][None, None, :]  # output_shape = (1, 1, 50176)

            np.save(f"{self.saved_file}/{video_name}.npy", wav_ft)
        except Exception:
            print("error:", wav_file_path)

    def logfbank_extract(self, wav_file_path, video_name):
        try:
            (rate, sig) = wav.read(wav_file_path)
            fbank_feat = logfbank(sig, rate)  # fbank_feat.shape = (3059,26)
            a = fbank_feat.flatten()
            single_vec_feat = a.reshape(1, -1)  # single_vec_feat.shape = (1,79534)
            np.save(f"{self.saved_file}/{video_name}.npy", single_vec_feat)
        except Exception:
            print("error:", wav_file_path)

    def opensmile_extract(self, wav_file_path, video_name):
        try:
            out = self.smile.process_file(wav_file_path)
            arr = np.array(out)
            np.save(f"{self.saved_file}/{video_name}.npy", arr)
        except Exception:
            print("error:", wav_file_path)

    @staticmethod
    def processed_files(save_to):
        processed_file = os.listdir(save_to)
        file_name = [item.replace(".npy", "") for item in processed_file]
        return file_name


def process(mode, saved_dir, *args):
    from tqdm import tqdm
    kwargs = {
        "parse_img_dir": False, "parse_aud_dir": True
    }
    processor = WaveProcessor(mode, saved_dir, *args, **kwargs)
    for idx in tqdm(range(len(processor))):
        processor[idx]


if __name__ == "__main__":
    # saved_dir_train = "../../../datasets/voice_data/voice_opensmile/train_data"
    # args_train = (
    #     "../../../datasets", None, "annotation/annotation_training.pkl", "raw_voice/trainingData")
    # process("opensmile", saved_dir_train, *args_train)

    # saved_dir_valid = "../../../datasets/voice_data/voice_opensmile/valid_data"
    # args_valid = (
    #     "../../../datasets", None, "annotation/annotation_validation.pkl", "raw_voice/validationData")
    # process("opensmile", saved_dir_valid, *args_valid)

    saved_dir_test = "../../../datasets/voice_data/voice_opensmile/test_data"
    args_test = (
        "../../../datasets", None, "annotation/annotation_test.pkl", "raw_voice/testData")
    process("opensmile", saved_dir_test, *args_test)

