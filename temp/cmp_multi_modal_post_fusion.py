import torch
import numpy as np
from dpcv.evaluation.metrics import compute_pcc, compute_ccc


class CmpPostFusion:
    def __init__(self, visual_data_path, audio_data_path, label_data_path, addition_data_path):
        self.visual_data = self.read_data(visual_data_path)
        self.audio_data = self.read_data(audio_data_path)
        self.label_data = self.read_data(label_data_path)
        self.addition = self.read_data(addition_data_path)

    def read_data(self, path):
        return torch.load(path)

    def compute(self):
        data = (self.visual_data + self.audio_data + self.addition) / 3
        ocean_acc = (1 - np.abs(data - self.label_data)).mean(axis=0)
        print(ocean_acc, ocean_acc.mean())

        pcc_dict, pcc_mean = compute_pcc(data, self.label_data)
        print(pcc_dict, pcc_mean)
        ccc_dict, ccc_mean = compute_ccc(data, self.label_data)
        print(ccc_dict, ccc_mean)


if __name__ == "__main__":
    import os; os.chdir("..")
    fusion = CmpPostFusion(
        visual_data_path="multi_modal_output/frame/pred.pkl",
        audio_data_path="multi_modal_output/audio/pred.pkl",
        label_data_path="multi_modal_output/audio/label.pkl",
        addition_data_path="multi_modal_output/face/pred.pkl"
    )
    fusion.compute()

"""
[0.91248983 0.91456544 0.9100043  0.9120678  0.9074899 ] 0.9113234
{'O': 0.6606, 'C': 0.7323, 'E': 0.6719, 'A': 0.5834, 'N': 0.6711} 0.6639
{'O': 0.5665, 'C': 0.6278, 'E': 0.5763, 'A': 0.4546, 'N': 0.5684} 0.5587

[0.91277313 0.9168967  0.9117387  0.91330695 0.9088075 ] 0.91270465
{'O': 0.6666, 'C': 0.7521, 'E': 0.6984, 'A': 0.6006, 'N': 0.6905} 0.6816
{'O': 0.5618, 'C': 0.6421, 'E': 0.5921, 'A': 0.462, 'N': 0.5734} 0.5663
"""