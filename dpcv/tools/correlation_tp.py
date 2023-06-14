import glob
import numpy as np
import torch
from itertools import permutations, combinations
from scipy.stats import pearsonr
import pickle



class ComCorrelationTP:
    def __init__(self, data_path):
        self.pred, self.label = self.read_pickle_data(data_path)

    @staticmethod
    def read_pickle_data(data_path):
        data_ls = glob.glob(f"{data_path}/*.pkl")
        video_pred, video_label = [], []
        for i in range(22):
            start = i * 100
            end = (i + 1) * 100
            output_ls, label_ls = [], []
            for data in data_ls[start: end]:
                output = torch.load(data)
                output_ls.append(output["video_frames_pred"])
                label_ls.append(output["video_label"])
            output = torch.cat(output_ls, dim=0).mean(dim=0).cpu().numpy().tolist()
            label = label_ls[0].cpu().numpy().tolist()
            video_pred.append(output)
            video_label.append(label)
        video_pred_arr = np.array(video_pred)
        video_label_arr = np.array(video_label)

        video_pred_dict, video_label_dict = {}, {}
        for idx, key in enumerate(["O", "C", "E", "A", "N"]):
            video_pred_dict[key] = video_pred_arr[:, idx]
            video_label_dict[key] = video_label_arr[:, idx]

        return video_pred_dict, video_label_dict
    
    def combination_pcc(self):
        pcc_dict = {}
        for (com_1, com_2) in combinations(self.pred.keys(), 2):
            vect_1 = self.pred[com_1]
            vect_2 = self.pred[com_2]
            pcc = pearsonr(vect_1, vect_2)
            com = com_1 + com_2
            pcc_dict[com] = pcc
        # print()
        save_data("udv_multi_pred_cor.pkl", pcc_dict)

        label_dict = {}
        for (com_1, com_2) in combinations(self.label.keys(), 2):
            vect_1 = self.label[com_1]
            vect_2 = self.label[com_2]
            pcc = pearsonr(vect_1, vect_2)
            com = com_1 + com_2
            label_dict[com] = pcc
        # print()
        save_data("udv_multi_label_cor.pkl", label_dict)




def save_data(name, data):
    with open(name, "wb") as f:
        pickle.dump(data, f)






if __name__ == "__main__":
    com = ComCorrelationTP("datasets/second_stage_tp/07_vat_face_video_level/test/animal_test")
    com.combination_pcc()