import pickle
from scipy.stats import pearsonr
from itertools import permutations, combinations
import numpy as np
import torch
import pickle
import glob


class ComCorrelation:

    def __init__(self, data_path, dataset="Chalearn16"):
        self.data = read_pickle_data(data_path)
        self.dataset = dataset

    def permute_pcc(self):
        if self.dataset == "Chalearn16":
            self.data.pop("interview")
        keys = self.data.keys()
        print()
        pcc_dict = {}
        for (com_1, com_2) in combinations(keys, 2):
            vect_1 = np.array([float(v) for v in self.data[com_1].values()])
            vect_2 = np.array([float(v) for v in self.data[com_2].values()])
            pcc = pearsonr(vect_1, vect_2)
            com = com_1[0] + com_2[0]
            pcc_dict[com] = pcc[0]

        save_data("cha_label_cor.pkl", pcc_dict)


class ComModelCorrelation:

    def __init__(self, data_path):
        self.data = self.read_pickle_data(data_path)

    @staticmethod
    def read_pickle_data(data_path):
        data_ls = glob.glob(f"{data_path}/*.pkl")
        pred_ls = []
        for data in data_ls:
            output = torch.load(data)
            pred = output["video_frames_pred"].mean(dim=0).cpu().numpy().tolist()
            pred_ls.append(pred)
        # print()
        pred_arr = np.array(pred_ls)
        pred_dict = {}
        for idx, key in enumerate(["O", "C", "E", "A", "N"]):
            pred_dict[key] = pred_arr[:, idx]
        return pred_dict

    def combination_pcc(self):
        pcc_dict = {}
        for (com_1, com_2) in combinations(self.data.keys(), 2):
            vect_1 = self.data[com_1]
            vect_2 = self.data[com_2]
            pcc = pearsonr(vect_1, vect_2)
            com = com_1 + com_2
            pcc_dict[com] = pcc
        print()
        save_data("cha_multi_pred_cor.pkl", pcc_dict)


def save_data(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)


def read_pickle_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


if __name__ == '__main__':

    # cor = ComCorrelation("datasets/annotation/annotation_test.pkl")
    # cor.permute_pcc()

    multi_pred = ComModelCorrelation("datasets/second_stage/vat/test")
    multi_pred.combination_pcc()

