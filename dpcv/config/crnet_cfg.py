import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
__C = cfg

__C.DATA_ROOT = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/"
__C.TRAIN_IMG_DATA = "ImageData/trainingData_face"
__C.TRAIN_AUD_DATA = "VoiceData/trainingData_244832"
__C.TRAIN_LABEL_DATA = "annotation_training.pkl"
__C.VALID_IMG_DATA = "ImageData/validationData_face"
__C.VALID_AUD_DATA = "VoiceData/validationData_244832"
__C.VALID_LABEL_DATA = "annotation_validation.pkl"

__C.TRAIN_BATCH_SIZE = 64  # 24
__C.VALID_BATCH_SIZE = 8  # 8
__C.NUM_WORKS = 0

__C.START_EPOCH = 0
__C.TRAIN_CLS_EPOCH = 10
__C.MAX_EPOCH = 40

__C.LR_INIT = 0.05
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 1e-4

__C.FACTOR = 0.1
__C.MILESTONE = [20, 30, 35]

# __C.RESUME = "../results/07-04_20-13/checkpoint_14.pkl"
__C.RESUME = None

__C.LOG_INTERVAL = 10

