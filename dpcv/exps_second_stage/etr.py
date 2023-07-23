from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import torch
import glob
import numpy as np


def data_collect(path):
    data_ls = glob.glob(f"{path}/*.pkl")
    data_ls = sorted(data_ls)
    data, ann = [], []
    for file in data_ls:
        val = torch.load(file)
        print()
        feat_mean = val["video_frames_feat"].mean(dim=0)
        label = val["video_label"]
        data.append(feat_mean)
        ann.append(label)
    train_data = torch.stack(data, dim=0).cpu().numpy()
    label_data = torch.stack(ann, dim=0).cpu().numpy()
    return train_data, label_data

if __name__ == "__main__":
    train_data, train_label = data_collect("datasets/stage_two/cr_net_extract/train")        
    test_data, test_label = data_collect("datasets/stage_two/cr_net_extract/test")
    print("fitting etr ...")
    reg = ExtraTreesRegressor(n_estimators=500, random_state=0).fit(train_data, train_label)
    print("finished etr training")
    pred = reg.predict(test_data)
    acc = (1 - np.abs(pred - test_label)).mean(axis=0)
    print(acc)
    
# [0.91009494 0.90845585 0.90609276 0.90867469 0.90601372] 0.9079