import os
import sys
from easydict import EasyDict

cfg = EasyDict()
_C = cfg

# test related setting  ------------------------------------------------------------------------------------------------
_C.TEST_ONLY = True
_C.WEIGHT = "../results/audio_visual_resnet/10-01_20-10/checkpoint_76.pkl"
_C.COMPUTE_PCC = True
_C.COMPUTE_CCC = True

# data set split config ------------------------------------------------------------------------------------------------
_C.OUTPUT_DIR = "../results/audio_visual_resnet"
_C.DATA_ROOT = "../datasets/"

_C.TRAIN_IMG_DATA = "image_data/train_data"
_C.VALID_IMG_DATA = "image_data/valid_data"
_C.TEST_IMG_DATA = "image_data/test_data"

_C.TRAIN_AUD_DATA = "voice_data/train_data"
_C.VALID_AUD_DATA = "voice_data/valid_data"
_C.TEST_AUD_DATA = "voice_data/test_data"

_C.TRAINVAL_IMG_DATA = ["image_data/train_data", "image_data/valid_data"]
_C.TRAINVAL_AUD_DATA = ["voice_data/trainingData", "voice_data/validationData"]

_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"
_C.TRAINVAL_LABEL_DATA = ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"]
# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 32  # 24
_C.VALID_BATCH_SIZE = 8  # 8
_C.SHUFFLE = True
_C.NUM_WORKS = 4
_C.START_EPOCH = 0
_C.MAX_EPOCH = 300

# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.0002
_C.FACTOR = 0.1
_C.MILESTONE = [100, 200]

# resume training ------------------------------------------------------------------------------------------------------

_C.RESUME = None  # "../results/08-07_bi-resnet_89.79/checkpoint_309.pkl"
_C.LOG_INTERVAL = 20