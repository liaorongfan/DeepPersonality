import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
__C = cfg

__C.DATA_ROOT = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/"
__C.RESULT_DIR = "../results"

__C.RESUME = None
__C.TRAIN_BATCH_SIZE = 32  # 24
__C.VALID_BATCH_SIZE = 8  # 8
__C.NUM_WORKS = 0
__C.LR_INIT = 0.0002
__C.FACTOR = 0.1
__C.MILESTONE = [300, 600]
__C.START_EPOCH = 0
__C.MAX_EPOCH = 900
__C.LOG_INTERVAL = 20
# __C.RESUME = "../results/07-01_22-28/checkpoint_23.pkl"
# __C.NORM_MEAN = [0.485, 0.456, 0.406]
# __C.NORM_STD = [0.229, 0.224, 0.225]
