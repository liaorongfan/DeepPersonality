import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg
_C.TEST_ONLY = False
_C.DATA_ROOT = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/"
_C.RESULT_DIR = "../results"

_C.TEST_ONLY = False
_C.WEIGHT = "../results/08-10_00-57/checkpoint_348.pkl"
_C.RESUME = None  # "../results/08-07_bi-resnet_89.79/checkpoint_309.pkl"
_C.TRAIN_BATCH_SIZE = 32  # 24
_C.VALID_BATCH_SIZE = 8  # 8
_C.NUM_WORKS = 4
_C.LR_INIT = 0.0002
_C.FACTOR = 0.1
_C.MILESTONE = [100, 200]
_C.START_EPOCH = 0
_C.MAX_EPOCH = 300
_C.LOG_INTERVAL = 20
