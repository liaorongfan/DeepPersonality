import os
import sys
import torchvision.transforms as transforms
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
__C = cfg

__C.DATA_ROOT = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/portrait"

__C.TRAIN_BATCH_SIZE = 24
__C.VALID_BATCH_SIZE = 8
__C.NUM_WORKS = 2
__C.LR_INIT = 0.01
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 1e-4
__C.FACTOR = 0.1
__C.MILESTONE = [20, 25]
__C.MAX_EPOCH = 30
__C.LOG_INTERVAL = 50

# norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万图像统计得来
# norm_std = [0.229, 0.224, 0.225]
# normTransform = transforms.Normalize(norm_mean, norm_std)

# cfg.transforms_train = transforms.Compose([
#     transforms.Resize(256),  # (256, 256) 区别； （256） 最短边256
#     transforms.CenterCrop(256),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     normTransform,
# ])
# cfg.transforms_valid = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     normTransform,
# ])
