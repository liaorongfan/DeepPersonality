"""
Temporary tool to extract wave sample features using librosa
"""
import os
import librosa
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData


class WaveProcess(VideoData):

    def __init__(self, save_to, *args):
        super(WaveProcess, self).__init__(*args)
        self.saved_file = save_to
        if not os.path.exists(save_to):
            os.makedirs(save_to)

    def __getitem__(self, idx):
        wav_file = self.img_dir_ls[idx] + ".wav"
        self.extract_save(wav_file)

    def extract_save(self, wav_file):
        wav_path = os.path.join(self.data_root, self.audio_dir, wav_file)
        try:
            wav_ft = librosa.load(wav_path, 3279)[0][None, None, :]  # output_shape = (1, 1, 50176)
            # wav_ft = librosa.load(wav_path, 1600)[0][None, None, :]  # output_shape = (1, 1, 244832)
            wav_temp = wav_ft
            if wav_ft.shape[-1] < 50176:
                wav_temp = np.zeros((1, 1, 50176))
                wav_temp[..., :wav_ft.shape[-1]] = wav_ft
            elif wav_ft.shape[-1] > 50176:
                wav_temp = wav_ft[..., :50176]
            np.save(f"{self.saved_file}/{wav_file}.npy", wav_temp)
        except Exception:
            print("error:", wav_file, wav_path)

    @staticmethod
    def processed_files(save_to):
        processed_file = os.listdir(save_to)
        file_name = [item.replace(".npy", "") for item in processed_file]
        return file_name


if __name__ == "__main__":
    from tqdm import tqdm
    saved_dir = "../../../datasets/VoiceData/testData_50176"
    args_train = ("../../../datasets", "ImageData/trainingData", "VoiceData/trainingData", "annotation_training.pkl")
    args_valid = (
        "../../../datasets", "ImageData/validationData", "VoiceData/validationData", "annotation_validation.pkl")
    args_test = (
        "../../../datasets", "ImageData/testData", "VoiceData/testData", "annotation_test.pkl")
    processor = WaveProcess(saved_dir, *args_test)
    for idx in tqdm(range(len(processor))):
        a = processor[idx]
    # test = processor[1]
    # print(test)
    # with open("../../../datasets/VoiceData/bad_case.txt") as f:
    #     files = [item.strip() for item in f.readlines()]

