"""
code modified form https://github.com/InnovArul/first-impressions
Thanks to the author
#
# pyAudioAnalysis should be installed first https://github.com/tyiannak/pyAudioAnalysis
# extractMFCCFeatures - to extract audio features from .wav files using pyAudioAnalysis toolkit
#      the featrures extracted are, as in https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
#
Note that the default setting of parameter 'deltas' is True which leads to 136 columns in xxx.wav_mt.csv file
which is not in line with the original paper
so modify to：
        feature_extraction(signal, sampling_rate, window, step, deltas=False)
"""
import subprocess
import os
from os import listdir
from os.path import isfile, join


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


# extractWavFile - to extract the wav audio file from mp4 files using ffmpeg
def extractWavFile(file_path, save_to):
    if isfile(file_path):
        if file_path.lower().endswith('.mp4'):
            path, filename = os.path.split(file_path)
            dest_path = os.path.abspath(save_to) + '_audio/'
            print(f"save file to {dest_path}")
            mkdir_p(dest_path)
            command = "ffmpeg -i " + file_path + " -ab 160k -ac 2 -ar 44100 -vn " + join(dest_path, filename) + ".wav"
            subprocess.call(command, shell=True)
    else:
        allfiles = [f for f in listdir(file_path) if (f != '.' and f != '..')]
        for any_file in allfiles:
            print(join(file_path, any_file))
            extractWavFile(join(file_path, any_file), save_to)


def extract_mfcc_features(file_path, exe_dir, save_to):
    if isfile(file_path):
        if file_path.lower().endswith('.wav'):
            path, filename = os.path.split(file_path)
            dest_path = os.path.abspath(save_to)
            mkdir_p(dest_path)
            (rate, sig) = audioBasicIO.read_audio_file(file_path)
            command = "python " + exe_dir + "/pyAudioAnalysis/audioAnalysis.py featureExtractionFile" \
                      " -i " + file_path + \
                      " -mw " + str(sig.shape[0] / float(rate) / 5.5) + \
                      " -ms " + str(sig.shape[0] / float(rate) / 5.5) + \
                      " -sw 0.050 " \
                      " -ss 0.050" \
                      " -o " + join(dest_path, filename)
            print(command)
            subprocess.call(command, shell=True)
    else:
        all_files = [f for f in listdir(file_path) if (f != '.' and f != '..')]
        for file in all_files:
            extract_mfcc_features(join(file_path, file), exe_dir, save_to)


def audio_preprocess_mfcc(root_path, exe_dir, save_to):
    wav_files_name = [file for file in os.listdir(root_path) if file.endswith(".wav")]
    wav_files_path = [os.path.join(root_path, file_name) for file_name in wav_files_name]
    for file_path in wav_files_path:
        extract_mfcc_features(file_path, exe_dir, save_to)

    subprocess.call("rm -f " + os.path.join(save_to, "*.npy"), shell=True)
    subprocess.call("rm -f " + os.path.join(save_to, "*.wav_st.csv"), shell=True)


if __name__ == "__main__":
    # pyAudioAnalysis should be installed first https://github.com/tyiannak/pyAudioAnalysis
    # or you can use the processed data provided by this program
    from pyAudioAnalysis import audioBasicIO

    data_root = "/home/ssd500/personality_data/raw_voice/validationData/"
    wav_file = "../datasets/VoiceData/validationData/i3s5XtbhSrE.004.wav"
    save_to = "/home/ssd500/personality_data/tem_voice/valid_data_mfcc"
    exe_dir = "../../../pyAudioAnalysis"

    # extract_mfcc_features(wav_file, exe_dir, save_to)
    audio_preprocess_mfcc(data_root, exe_dir, save_to)