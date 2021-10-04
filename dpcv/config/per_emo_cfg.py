import os
import sys
from easydict import EasyDict

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg

# data set split config ------------------------------------------------------------------------------------------------
_C.DATA_ROOT = "../datasets/"
_C.TRAIN_IMG_DATA = "image_data/train_data_face"
_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VA_DATA = "va_data/cropped_aligned"
_C.VA_TRAIN_LABEL = "va_data/va_label/VA_Set/Train_Set"
_C.VA_VALID_LABEL = "va_data/va_label/VA_Set/Validation_Set"
_C.VALID_IMG_DATA = "image_data/valid_data_face"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_IMG_DATA = "image_data/test_data"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"
# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 1  # 24
_C.VALID_BATCH_SIZE = 1  # 8
_C.SHUFFLE = True
_C.NUM_WORKS = 4
_C.START_EPOCH = 0
_C.MAX_EPOCH = 100
# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.05
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 0.0005
_C.FACTOR = 0.1
_C.MILESTONE = [25, 70, 90]

_C.PRE_TRAINED_MODEL = None
_C.RESUME = None  # "../results/10-04_11-22/checkpoint_21.pkl"

_C.LOG_INTERVAL = 20
