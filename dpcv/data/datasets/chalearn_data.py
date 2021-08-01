"""
code modified form https://github.com/zishansami102/First-Impression
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import pickle
import glob
import numpy as np
from random import shuffle
import os


class ChaLearnImage(Dataset):

    def __init__(self, data_base, mode="train", transform=None):
        assert mode in ["train", "valid"], "mode should be one 'train' or 'valid'"
        self.data_base = data_base
        self.transform = transform
        self.mode = mode
        self._img_list = []
        self._ano_list = []
        self.parse_annotation()

    def __getitem__(self, index):
        image_name, ano = self._img_list[index], self._ano_list[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            ano = torch.as_tensor(ano)
        sample = {"image": image, "label": ano}
        return sample

    def __len__(self):
        return len(self._img_list)

    def parse_annotation(self):
        data_file_path = os.path.join(self.data_base, "annotation_training.pkl")
        if self.mode == "valid":
            data_file_path = os.path.join(self.data_base, "annotation_validation.pkl")

        with open(data_file_path, 'rb') as fo:
            pickle_file = pickle.load(fo, encoding="latin1")
        df = pd.DataFrame(pickle_file)
        df.reset_index(inplace=True)
        del df["interview"]
        df.columns = [
            "VideoName",
            "ValueExtraversion", "ValueNeuroticism", "ValueAgreeableness",
            "ValueConscientiousness", "ValueOpenness"
        ]
        temp_img_list = []
        temp_ano_list = []
        data_dir = os.path.join(self.data_base, 'ImageData/trainingData/')
        if self.mode == "valid":
            data_dir = os.path.join(self.data_base, 'ImageData/validationData/')

        for i in range(len(df)):
            file_list = glob.glob(data_dir + (df['VideoName'].iloc[i]).split('.mp4')[0] + '/*.jpg')
            temp_img_list += file_list
            temp_ano_list += [
                np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)
            ] * len(file_list)

        temp = list(zip(temp_img_list, temp_ano_list))
        shuffle(temp)
        self._img_list, self._ano_list = zip(*temp)

    def show_image(self, index):
        image_name = self._img_list[index]
        image = Image.open(image_name).convert('RGB')
        return image


def set_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def make_data_loader(cfg, mode):
    assert (mode in ["train", "valid"]), " 'mode' only supports 'train' and 'valid'"
    transforms = set_transform_op()
    dataset = ChaLearnImage(cfg.DATA_ROOT, mode, transforms)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKS
    )
    return data_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trans = set_transform_op()
    cha_data = ChaLearnImage("../../../datasets", mode="train", transform=trans)
    print(len(cha_data))
    print(cha_data[2])
    for i in range(4, 10):
        sample = cha_data[i]
        img, label = sample["image"], sample["label"]
        print(img.shape, label.shape)
        print(img.dtype, label.dtype)
        image = cha_data.show_image(i)
        plt.imshow(image)
        plt.show()

    data_loader = DataLoader(dataset=cha_data, batch_size=4, shuffle=True)
    for i, batch in enumerate(data_loader):
        if i >= 3:
            break
        batch_img, batch_anno = batch["image"], batch["label"]
        print(batch_img.shape, batch_anno.shape)
