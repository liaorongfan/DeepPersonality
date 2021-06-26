import os
import torch
import random
import numpy as np
# import psutil


def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


def check_data_dir(path_tmp):
    assert os.path.exists(path_tmp), \
        f"\n\n路径不存在，当前变量中指定的路径是：\n{os.path.abspath(path_tmp)}\n请检查相对路径的设置，或者文件是否存在"
