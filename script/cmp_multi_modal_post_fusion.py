import torch
import numpy as np
from dpcv.evaluation.metrics import compute_pcc, compute_ccc


class CmpPostFusion:
    def __init__(self, visual_data_path, audio_data_path, label_data_path, addition_data_path=None):
        self.visual_data = self.read_data(visual_data_path)
        self.audio_data = self.read_data(audio_data_path)
        self.label_data = self.read_data(label_data_path)
        self.addition = None
        self.num_modal = 2
        if addition_data_path:
            self.addition = self.read_data(addition_data_path)
            self.num_modal += 1

    def read_data(self, path):
        return torch.load(path)

    def compute(self):
        max_len = min(len(self.visual_data), len(self.audio_data))
        self.visual_data = self.visual_data[:max_len, :]
        self.audio_data = self.audio_data[:max_len, :]
        self.label_data = self.label_data[:max_len, :]

        if self.addition:
            data = (self.visual_data + self.audio_data + self.addition) / self.num_modal
        else:
            data = (self.visual_data + self.audio_data) / self.num_modal

        mse = np.square(data - self.label_data).mean(axis=0)
        print("mse: ", mse, mse.mean())

        ocean_acc = (1 - np.abs(data - self.label_data)).mean(axis=0)
        print("acc: ", ocean_acc, ocean_acc.mean())

        pcc_dict, pcc_mean = compute_pcc(data, self.label_data)
        print("pcc: ", pcc_dict, pcc_mean)
        ccc_dict, ccc_mean = compute_ccc(data, self.label_data)
        print("ccc: ", ccc_dict, ccc_mean)


if __name__ == "__main__":
    import os; os.chdir("..")
    # fusion = CmpPostFusion(
    #     visual_data_path="multi_modal_output/true_personality/talk/frame/pred.pkl",
    #     audio_data_path="multi_modal_output/true_personality/talk/audio/pred.pkl",
    #     label_data_path="multi_modal_output/true_personality/talk/audio/label.pkl",
    #     addition_data_path="multi_modal_output/true_personality/talk/face/pred.pkl"
    # )
    fusion = CmpPostFusion(
        visual_data_path="tmp/impresssion/frame/pred.pkl",
        audio_data_path="tmp/impresssion/audio/pred.pkl",
        label_data_path="tmp/impresssion/audio/label.pkl"
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

"""
animal: 
    mse:  [0.93170875 0.75650316 1.6238841  1.0261804  1.1671154 ] 1.1010784
    acc:  [ 0.2805941   0.33851784 -0.03436301  0.28129753  0.04835478] 0.18288025
    pcc:  {'O': -0.0727, 'C': -0.0916, 'E': 0.2207, 'A': 0.1429, 'N': 0.1795} 0.0758
    ccc:  {'O': -0.0182, 'C': -0.0227, 'E': 0.0268, 'A': 0.0236, 'N': 0.0091} 0.0037
ghost:
    mse:  [0.94146466 0.671024   1.6057849  0.9854744  1.1656427 ] 1.0738782
    acc:  [ 0.28378037  0.37203065 -0.02701496  0.29230556  0.0454604 ] 0.19331239
    pcc:  {'O': -0.0789, 'C': 0.2861, 'E': 0.2533, 'A': 0.3166, 'N': 0.1982} 0.1951
    ccc:  {'O': -0.0213, 'C': 0.0491, 'E': 0.0252, 'A': 0.0637, 'N': 0.0069} 0.0247        
lego:
    mse:  [0.92159975 0.63222516 1.37113    0.9128006  1.1516751 ] 0.99788606
    acc:  [0.30695382 0.37930268 0.06725857 0.28737295 0.02967892] 0.2141134
    pcc:  {'O': -0.1458, 'C': 0.2434, 'E': 0.2408, 'A': 0.3507, 'N': 0.26} 0.1898
    ccc:  {'O': -0.0304, 'C': 0.0628, 'E': 0.0314, 'A': 0.0163, 'N': 0.0} 0.016
talk:
    mse:  [0.961764   0.61027384 1.3815945  0.91071665 1.1514109 ] 1.003152
    acc:  [0.29116476 0.37448215 0.03584053 0.2908995  0.02993069] 0.20446353
    pcc:  {'O': -0.2427, 'C': 0.413, 'E': 0.301, 'A': 0.2607, 'N': 0.152} 0.1768
    ccc:  {'O': -0.0695, 'C': 0.098, 'E': 0.0373, 'A': 0.0553, 'N': 0.0003} 0.0243
"""