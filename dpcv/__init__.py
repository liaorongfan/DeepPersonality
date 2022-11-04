import torch
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
# optionally print the sys.path for debugging)
# print("in _ _init_ _.py sys.path:\n ",sys.path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
