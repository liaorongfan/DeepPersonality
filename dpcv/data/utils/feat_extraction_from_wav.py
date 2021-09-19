from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pickle
import numpy as np
from random import shuffle
import glob
import pandas as pd
import sys


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f, encoding='latin1')
        df = pd.DataFrame(pickle_data)
        df.reset_index(inplace=True)
        del df['interview']
        df.columns = ["VideoName", "ValueExtraversion", "ValueNeuroticism", "ValueAgreeableness",
                      "ValueConscientiousness", "ValueOpenness"]
    return df


def process_wav(wav_file):
    (rate, sig) = wav.read(wav_file)
    fbank_feat = logfbank(sig, rate)  # fbank_feat.shape = (3059,26)
    a = fbank_feat.flatten()
    single_vec_feat = a.reshape(1, -1)  # single_vec_feat.shape = (1,79534)
    return single_vec_feat


# def _float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


df = load_pickle('../../../datasets/unzippedData/annotation_training.pkl')
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist = glob.glob('VoiceData/trainingData/' + (df['VideoName'].iloc[i]).split('.mp4')[0] + '.wav')
    addrs += filelist
    labels += [np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)] * 100

c = list(zip(addrs, labels))
shuffle(c)

# ######################################### write TFRecords for train data ##########################################
# train_addrs, train_labels = zip(*c)
# # train_addrs, train_labels = addrs , labels
# train_filename = 'train_audio_full.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(train_filename)
# for i in range(len(train_addrs)):
#     # print how many audio are saved every 1000 images
#     if not i % 1000:
#         print('Train data: {}/{}'.format(i, len(train_addrs)))
#         sys.stdout.flush()
#     # Load the audio
#     audio = process_wav(train_addrs[i])
#     label = train_labels[i]
#     # Create a feature
#     feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
#                'train/audio': _bytes_feature(tf.compat.as_bytes(audio.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
#
# writer.close()
# sys.stdout.flush()
#
# print(len(train_addrs), "training audio files saved.. ")

# ######################################### write TFRecords for validate data #######################################
df = load_pickle('../../../datasets/annotation/annotation_validation.pkl')
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist = glob.glob('VoiceData/validationData/' + (df['VideoName'].iloc[i]).split('.mp4')[0] + '.wav')
    addrs += filelist
    labels += [np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)] * 100

c = list(zip(addrs, labels))
shuffle(c)
val_addrs, val_labels = zip(*c)

# val_addrs, val_labels = addrs , labels
# val_filename = 'val_audio_full.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(val_filename)
# for i in range(len(val_addrs)):
#     # print how many audio are saved every 1000 images
#     if not i % 1000:
#         print('val data: {}/{}'.format(i, len(val_addrs)))
#         sys.stdout.flush()
#     # Load the audio
#     audio = process_wav(val_addrs[i])
#     label = val_labels[i]
#     # Create a feature
#     feature = {'val/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
#                'val/audio': _bytes_feature(tf.compat.as_bytes(audio.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
#
# writer.close()
# sys.stdout.flush()
#
# print(len(val_addrs), "val audio files saved.. ")
