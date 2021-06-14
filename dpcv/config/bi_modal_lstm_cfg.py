import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
__C = cfg

__C.DATA_ROOT = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/"

__C.TRAIN_BATCH_SIZE = 4  # 24
__C.VALID_BATCH_SIZE = 4  # 8
__C.NUM_WORKS = 0
__C.LR_INIT = 0.05
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 5e-4
__C.FACTOR = 0.1
__C.MILESTONE = [1, 2, ]
__C.MAX_EPOCH = 4
__C.LOG_INTERVAL = 50
