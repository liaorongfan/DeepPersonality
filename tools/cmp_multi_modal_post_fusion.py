import torch
import os
import numpy as np
from dpcv.evaluation.metrics import compute_pcc, compute_ccc
import logging
import argparse


class CmpPostFusion:
    def __init__(self, visual_data_path, audio_data_path, label_data_path, addition_data_path=None):
        logger.info(f" ================ {os.path.dirname(visual_data_path)} ================")
        self.visual_data = self.read_data(visual_data_path)
        self.audio_data = self.read_data(audio_data_path)
        self.label_data = self.read_data(label_data_path)
        self.addition = []
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

        if len(self.addition) > 0:
            data = (self.visual_data + self.audio_data + self.addition) / self.num_modal
        else:
            data = (self.visual_data + self.audio_data) / self.num_modal

        mse = np.square(data - self.label_data).mean(axis=0)
        logger.info(f"mse: {mse}, {mse.mean()}")

        ocean_acc = (1 - np.abs(data - self.label_data)).mean(axis=0)
        logger.info(f"acc: {ocean_acc}, {ocean_acc.mean()}")

        pcc_dict, pcc_mean = compute_pcc(data, self.label_data)
        logger.info(f"pcc: {pcc_dict}, {pcc_mean}")
        ccc_dict, ccc_mean = compute_ccc(data, self.label_data)
        logger.info(f"ccc: {ccc_dict}, {ccc_mean}")


if __name__ == "__main__":
    # import os; os.chdir("..")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(message)s',
    )  # define the format when print on screen

    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s- %(message)s')  # define the format when recorded in files
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description="post fusion")
    parser.add_argument("-v", "--visual", type=str)
    parser.add_argument("-a", "--audio", type=str)
    parser.add_argument("-o", "--other", default="")
    parser.add_argument("-l", "--label", type=str)
    args = parser.parse_args()

    # fusion = CmpPostFusion(
    #     visual_data_path="tmp/impresssion/frame/pred.pkl",
    #     audio_data_path="tmp/impresssion/audio/pred.pkl",
    #     label_data_path="tmp/impresssion/audio/label.pkl"
    # )

    fusion = CmpPostFusion(
        visual_data_path=args.visual,
        audio_data_path=args.audio,
        label_data_path=args.label,
        addition_data_path=args.other,
    )

    fusion.compute()
