from random import shuffle
import glob
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
from PIL import Image
import cv2
import pickle


def load_image(addr):
    img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# df.loc[df['column_name'] == some_value]
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f, encoding='latin1')
        df = pd.DataFrame(pickle_data)
        df.reset_index(inplace=True)
        del df['interview']
        df.columns = ["VideoName", "ValueExtraversion", "ValueNeuroticism", "ValueAgreeableness",
                      "ValueConscientiousness", "ValueOpenness"]
    return df


# ############################################### write tf ########################################################
df = load_pickle('annotation_training.pkl')
NUM_VID = len(df)
addrs = []
labels = []
sample_idx = [1121, 4156, 4700]
# for i in range(NUM_VID):
for i in sample_idx:
    filelist = glob.glob('ImageData/trainingData/' + (df['VideoName'].iloc[i]).split('.mp4')[0] + '/*.jpg')
    addrs += filelist
    labels += [np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)] * 100

c = list(zip(addrs, labels))
shuffle(c)
train_addrs, train_labels = zip(*c)
train_filename = 'train_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

print(len(train_addrs), "training images saved.. ")
#
# df = load_pickle('annotation_validation.pkl')
# NUM_VID = len(df)
# addrs = []
# labels = []
# for i in range(NUM_VID):
#     filelist = glob.glob('ImageData/validationData/' + (df['VideoName'].iloc[i]).split('.mp4')[0] + '/*.jpg')
#     addrs += filelist
#     labels += [np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)] * 100
#
# c = list(zip(addrs, labels))
# shuffle(c)
# val_addrs, val_labels = zip(*c)
#
# val_filename = 'val_full.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(val_filename)
#
# for i in range(len(val_addrs)):
#     # print how many images are saved every 1000 images
#     if not i % 1000:
#         print('Val data: {}/{}'.format(i, len(val_addrs)))
#         sys.stdout.flush()
#     # Load the image
#     img = load_image(val_addrs[i])
#     label = val_labels[i].astype(np.float32)
#     feature = {'val/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
#                'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
#
# writer.close()
# sys.stdout.flush()
#
# print(len(train_addrs), "training images saved.. ")
# print(len(val_addrs), "validation images saved.. ")
