import glob
import torch
import os
from itertools import permutations, combinations
from scipy.stats import pearsonr
import pickle


class CollectSingleTrait:

    def __init__(self, data_path="datasets/second_stage/vat_single"):
        self.data = self.read_pickle_data(data_path)

    @staticmethod
    def read_pickle_data(data_path):
        dirs = os.listdir(data_path)
        data_dict = {}
        for d in dirs:
            data_ls = []
            data_files = glob.glob(f"{data_path}/{d}/test/*.pkl")
            for file in data_files:
                data = torch.load(file)
                data = data["video_frames_pred"].mean(dim=0).cpu().item()
                data_ls.append(data)
            data_dict[d[-1]] = data_ls
        return data_dict

    def combination_pcc(self):
        pcc_dict = {}
        for (com_1, com_2) in combinations(self.data.keys(), 2):
            vect_1 = self.data[com_1]
            vect_2 = self.data[com_2]
            pcc = pearsonr(vect_1, vect_2)
            com = com_1 + com_2
            pcc_dict[com] = pcc
        print()
        save_data("cha_single_pred_cor.pkl", pcc_dict)


def save_data(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)



if __name__ == "__main__":
    col = CollectSingleTrait()
    col.combination_pcc()