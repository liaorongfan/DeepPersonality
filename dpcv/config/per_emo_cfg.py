import os
import sys
from easydict import EasyDict

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg


# test related setting  ------------------------------------------------------------------------------------------------
_C.TEST_ONLY = False
_C.WEIGHT = "../results/per_emo/11-03_01-42/checkpoint_782.pkl"
_C.COMPUTE_PCC = True
_C.COMPUTE_CCC = True
# data set split config ------------------------------------------------------------------------------------------------
_C.DATA_ROOT = "../datasets/"
_C.TRAIN_IMG_DATA = "image_data/train_data_face"
_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VA_DATA = "va_data/cropped_aligned"
_C.VA_TRAIN_LABEL = "va_data/va_label/VA_Set/Train_Set"
_C.VA_VALID_LABEL = "va_data/va_label/VA_Set/Validation_Set"
_C.VALID_IMG_DATA = "image_data/valid_data_face"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_IMG_DATA = "image_data/test_data_face"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"
# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 8  # 24
_C.VALID_BATCH_SIZE = 4  # 8
_C.SHUFFLE = True
_C.NUM_WORKS = 4
_C.START_EPOCH = 0
_C.MAX_EPOCH = 600
# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.001
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 0.0005
_C.FACTOR = 0.5
_C.MILESTONE = [450, 550]

_C.PRE_TRAINED_MODEL = None
_C.RESUME = None
# _C.RESUME = "../results/per_emo/11-03_01-42/checkpoint_782.pkl"
_C.LOG_INTERVAL = 20
_C.OUTPUT_DIR = "../results/per_emo"
