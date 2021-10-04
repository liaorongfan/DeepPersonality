import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
__C = cfg

__C.DATA_ROOT = "DeepPersonality/datasets/"
__C.TRAIN_IMG_DATA = "image_data/training_data_01"
__C.TRAIN_AUD_DATA = "voice_data/training_data_244832"
__C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
__C.VALID_IMG_DATA = "image_data/validation_data_01"
__C.VALID_AUD_DATA = "voiceData/validation_data_244832"
__C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"

__C.TRAIN_BATCH_SIZE = 32
__C.VALID_BATCH_SIZE = 32
__C.NUM_WORKS = 4
__C.BETA_1 = 0.5
__C.BETA_2 = 0.999
__C.START_EPOCH = 0
__C.TRAIN_CLS_EPOCH = 20
__C.MAX_EPOCH = 50

__C.LR_INIT = 0.002
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 0.005

__C.FACTOR = 0.1
__C.MILESTONE = [30, 40]

# __C.RESUME = "../results/07-04_20-13/checkpoint_14.pkl"
__C.RESUME = None

__C.LOG_INTERVAL = 20

