import subprocess
import os
import zipfile


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
