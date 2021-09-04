import subprocess
import os
import zipfile


def video2wave_tst(data_dir, save_dir="VoiceData/testData/"):
    for video in os.listdir(data_dir):
        file_name = video.split(".mp4")[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        command = f"ffmpeg -i {data_dir}/{video} -ab 320k -ac 2 -ar 44100 -vn {save_dir}/{file_name}.wav"
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    video2wave_tst("chalearn_test_video")
