import glob
import os.path
import torch
from tqdm import tqdm
from sklearn import linear_model
from collections import defaultdict
import numpy as np
from dpcv.evaluation.metrics import compute_ccc


class LinearRegression:

    def __init__(self, data_root):
        self.data_root = data_root
        self.traits = ["O", "C", "E", "A", "N"]
        self.train_data = self.collect_data("train")
        self.test_data = self.collect_data("valid")
        self.linear_models = self.get_models()
        self.predict_results = None

    def get_models(self):
        models = {}
        for t in self.traits:
            models[t] = linear_model.LinearRegression()
        return models

    def collect_data(self, split):
        data_dir = os.path.join(self.data_root, split)
        data_ls = glob.glob(f"{data_dir}/*")
        sample_x = defaultdict(list)
        sample_y = defaultdict(list)

        for data_path in tqdm(data_ls):
            data = torch.load(data_path)
            frame_preds = data["video_frames_pred"]
            frame_target = data["video_label"]
            if len(frame_preds) < 400:
                continue
            for i, t in enumerate(self.traits):
                sample_x[t].append(frame_preds.numpy()[:400, i].tolist())
                sample_y[t].append(frame_target.numpy()[i].tolist())

        return sample_x, sample_y

    def fit_models(self):
        x, y = self.train_data
        for t in self.traits:
            x_arr, y_arr = np.array(x[t]), np.array(y[t])
            self.linear_models[t].fit(x_arr, y_arr)

    def predict(self):
        x_t, y_t = self.test_data
        y_pred_dict = {}
        for t in self.traits:
            x_t_arr = np.array(x_t[t])
            y_pred = self.linear_models[t].predict(x_t_arr)
            y_pred_dict[t] = y_pred

        self.predict_results = y_pred_dict
        # self.linear_model.fit(x, y)
        # pred_t = self.linear_model.predict(x_t)
        print()

    def compute_metrics(self):
        _, target = self.test_data
        results, targets = [], []
        for t in self.traits:
            results.append(self.predict_results[t])
            targets.append(target[t])
        results, targets = np.array(results).T, np.array(targets).T
        ccc_dict, mean = compute_ccc(results, targets)
        print(ccc_dict, mean)
        self.record_results(ccc_dict, mean)

    def record_results(self, ccc, mean):
        data_dir_name = os.path.basename(self.data_root)
        with open(f"{data_dir_name}_linear.txt", 'w') as f:
            f.write(str(ccc) + " mean: " + str(mean))

    def run(self):
        self.fit_models()
        self.predict()
        self.compute_metrics()


if __name__ == '__main__':
    data_roots = glob.glob("datasets/second_stage/*")
    for dr in data_roots:
        LinearRegression(dr).run()
        # linear_fusion.run()
