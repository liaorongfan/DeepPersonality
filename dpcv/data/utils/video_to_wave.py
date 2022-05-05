import glob
import subprocess
import os
from tqdm import tqdm
import zipfile
import librosa
from pathlib import Path
import numpy as np


class ChaLearn16AudioExtract:
    @staticmethod
    def video2wave_train(zipfile_dir):
        # Running a loop through all the zipped training file to extract all .wav audio files
        for i in range(1, 76):
            if i < 10:
                zipfilename = 'training80_0' + str(i) + '.zip'
            else:
                zipfilename = 'training80_' + str(i) + '.zip'
            # Accessing the zipfile i
            archive = zipfile.ZipFile(zipfile_dir + zipfilename, 'r')
            zipfilename = zipfilename.split('.zip')[0]
            # archive.extractall('unzippedData/'+zipfilename)
            for file_name in archive.namelist():
                file_name = (file_name.split('.mp4'))[0]
                try:
                    if not os.path.exists('../../../datasets/VoiceData/trainingData/'):
                        os.makedirs('../../../datasets/VoiceData/trainingData/')
                except OSError:
                    print('Error: Creating directory of data')
                command = "ffmpeg -i unzippedData/{}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/trainingData/{}.wav"\
                    .format(zipfilename, file_name, file_name)
                subprocess.call(command, shell=True)

    @staticmethod
    def video2wave_val(zipfile_dir):
        for i in range(1, 26):
            if i < 10:
                zipfilename = 'validation80_0' + str(i) + '.zip'
            else:
                zipfilename = 'validation80_' + str(i) + '.zip'
            # Accessing the zipfile i
            archive = zipfile.ZipFile(zipfile_dir + zipfilename, 'r')
            zipfilename = zipfilename.split('.zip')[0]
            # archive.extractall('unzippedData/'+zipfilename)
            for file_name in archive.namelist():
                file_name = (file_name.split('.mp4'))[0]
                try:
                    if not os.path.exists('../../../datasets/VoiceData/validationData/'):
                        os.makedirs('../../../datasets/VoiceData/validationData/')
                except OSError:
                    print('Error: Creating directory of data')
                command = "ffmpeg -i unzippedData/{}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/validationData/{}.wav"\
                    .format(zipfilename, file_name, file_name)
                subprocess.call(command, shell=True)

    @staticmethod
    def video2wave_tst(data_dir):
        for video in os.listdir(data_dir):
            file_name = video.split(".mp4")[0]
            if not os.path.exists("VoiceData/testData"):
                os.makedirs("VoiceData/testData")
            command = f"ffmpeg -i {data_dir}/{video} -ab 320k -ac 2 -ar 44100 -vn VoiceData/testData/{file_name}.wav"
            subprocess.call(command, shell=True)


def chalearn21_audio_extract_ffmpeg(dir_path):
    path = Path(dir_path)
    mp4_ls = path.rglob("./*/*.mp4")
    for mp4 in mp4_ls:
        parent_dir, name = mp4.parent, mp4.stem
        cmd = f"ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav"
        subprocess.call(cmd, shell=True)


def chalearn21_audio_process(dir_path):
    wav_path_ls = glob.glob(f"{dir_path}/*/*.wav")
    for wav_path in tqdm(wav_path_ls):
        try:
            wav_ft = librosa.load(wav_path, 16000)[0][None, None, :]  # output_shape = (1, 1, 244832)
            wav_name = wav_path.replace(".wav", ".npy")
            np.save(wav_name, wav_ft)
        except Exception:
            print("error:", wav_path)


if __name__ == "__main__":
    dir_path = "/home/rongfan/05-personality_traits/DeepPersonality/datasets/chalearn2021/test/talk_test"
    # chalearn21_audio_extract_ffmpeg(dir_path)
    chalearn21_audio_process(dir_path)
