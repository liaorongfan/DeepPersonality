import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg
# if only test or resume training or train from scratch ----------------------------------------------------------------
_C.TEST_ONLY = True
_C.WEIGHT = "../results/cr_net/10-22_01-38/checkpoint_85.pkl"
# __C.RESUME = "../results/07-04_20-13/checkpoint_14.pkl"
_C.RESUME = None
# data related config --------------------------------------------------------------------------------------------------
_C.DATA_ROOT = "DeepPersonality/datasets/"
_C.OUTPUT_DIR = "../results/cr_net"
_C.TRAIN_IMG_DATA = "image_data/training_data"
_C.TRAIN_AUD_DATA = "voice_data/training_data"
_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VALID_IMG_DATA = "image_data/validation_data"
_C.VALID_AUD_DATA = "voiceData/validation_data"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
# data loader related config -------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 32
_C.VALID_BATCH_SIZE = 32
_C.NUM_WORKS = 4
# optimizer setting ----------------------------------------------------------------------------------------------------
# optimizer one
_C.BETA_1 = 0.5
_C.BETA_2 = 0.999
# optimizer two
_C.LR_INIT = 0.002
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 0.005
# learning rate scheduler
_C.FACTOR = 0.1
_C.MILESTONE = [80, 90]
# training epoch setting -----------------------------------------------------------------------------------------------
_C.START_EPOCH = 0
_C.TRAIN_CLS_EPOCH = 30
_C.MAX_EPOCH = 100
# logging related setting ----------------------------------------------------------------------------------------------
_C.LOG_INTERVAL = 20

