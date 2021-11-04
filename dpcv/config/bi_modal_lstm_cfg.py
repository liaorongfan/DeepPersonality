import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg

# test related setting  ------------------------------------------------------------------------------------------------
_C.TEST_ONLY = False
_C.WEIGHT = "../results/bi_modal_lstm/11-04_22-46/checkpoint_21.pkl"
_C.COMPUTE_PCC = True
_C.COMPUTE_CCC = True

# data set split config ------------------------------------------------------------------------------------------------
_C.OUTPUT_DIR = "../results/bi_modal_lstm"
_C.DATA_ROOT = "../datasets/"

_C.TRAIN_IMG_DATA = "image_data/train_data_face"
_C.VALID_IMG_DATA = "image_data/valid_data_face"
_C.TEST_IMG_DATA = "image_data/test_data_face"

_C.TRAIN_AUD_DATA = "voice_data/voice_mfcc/train_data_mfcc"
_C.VALID_AUD_DATA = "voice_data/voice_mfcc/valid_data_mfcc"
_C.TEST_AUD_DATA = "voice_data/voice_mfcc/test_data_mfcc"

_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"

# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 64
_C.VALID_BATCH_SIZE = 32
_C.NUM_WORKS = 4
_C.START_EPOCH = 0
_C.MAX_EPOCH = 200

# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.05
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 5e-4
_C.FACTOR = 1
_C.MILESTONE = [120, 150, 180]

# resume training ------------------------------------------------------------------------------------------------------
_C.RESUME = None
_C.LOG_INTERVAL = 20
