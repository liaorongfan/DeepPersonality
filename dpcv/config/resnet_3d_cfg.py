import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg

# test related setting  ------------------------------------------------------------------------------------------------
_C.TEST_ONLY = True
_C.WEIGHT = "../results/resnet3d/11-08_21-00/checkpoint_94.pkl"
_C.COMPUTE_PCC = True
_C.COMPUTE_CCC = True

# data set split config ------------------------------------------------------------------------------------------------
_C.DATA_ROOT = "../datasets/"
_C.OUTPUT_DIR = "../results/resnet3d"
_C.TRAIN_IMG_DATA = "image_data/train_data"
_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VALID_IMG_DATA = "image_data/valid_data"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_IMG_DATA = "image_data/test_data"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"
_C.TRAINVAL_IMG_DATA = ["image_data/train_data", "image_data/valid_data"]
_C.TRAINVAL_LABEL_DATA = ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"]
# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 8  # 24
_C.VALID_BATCH_SIZE = 8  # 8
_C.SHUFFLE = True
_C.NUM_WORKERS = 4  # for debug in pycharm set to 0
_C.START_EPOCH = 0
_C.MAX_EPOCH = 100
# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.01
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 0.0005
_C.FACTOR = 0.1
_C.MILESTONE = [80, 90]

_C.RESUME = None

_C.LOG_INTERVAL = 10
