"""
code modified form https://github.com/miguelmore/personality
"""
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PortraitDataset(Dataset):

    # 30935 total
    total_train = 28152
    total_test = 2783

    def __init__(self, data_path, mode="train", transforms=None):
        self.data_path = data_path
        # self.mode = mode
        self.data_set = self._load_dataset(mode)
        self.transforms = transforms

    def __getitem__(self, idx):
        image, annotation = self.data_set['X'][idx],  self.data_set['Y'][idx]
        if self.transforms is not None:
            image = self.transforms(image)
            annotation = torch.as_tensor(annotation, dtype=image.dtype)
        sample = {"image": image, "label": annotation}
        return sample

    def __len__(self):
        return len(self.data_set["X"])

    def get_image(self, idx):
        return self.data_set['X'][idx]

    def _load_dataset(self, mode):
        if mode == "train":
            data_train = {'X': [], 'Y': []}
            for i in range(1, 4):
                x, y = self._read_pickle('train_clselfie_v4_{}.pickle'.format(i), show=True)
                data_train['X'].extend(x)
                data_train['Y'].extend(y)
            data_train['X'] = np.array(data_train['X'])
            data_train['Y'] = np.array(data_train['Y'])
            print('\nTotal Train Data X:', data_train['X'].shape, 'Y:', data_train['Y'].shape)
            return data_train
        elif mode == "test":
            data_test = {'X': [], 'Y': []}
            x, y = self._read_pickle('test_clselfie_v4.pickle', show=True)
            data_test['X'].extend(x)
            data_test['Y'].extend(y)

            data_test['X'] = np.array(data_test['X'])
            data_test['Y'] = np.array(data_test['Y'])
            print('Total Test  Data X:', data_test['X'].shape, 'Y:', data_test['Y'].shape)
            return data_test
        else:
            raise ValueError("data loading only support mode of 'train' or 'test' ")

    def _read_pickle(self, name, show=False):
        path = os.path.join(self.data_path, name)
        pic = pickle.load(open(path, "rb"))
        x = np.array(pic['X'])
        y = np.array(pic['Y'])
        if show:
            print('X-', x.shape)
            print('Y-', y.shape)
        return x, y


def set_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def make_data_loader(cfg, mode):
    assert (mode in ["train", "test"]), " 'mode' only supports 'train' and 'test'"

    transforms = set_transform_op()
    dataset = PortraitDataset(cfg.flower_data_root, mode, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    return data_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_path = "/home/rongfan/11-personality_traits/DeepPersonality/datasets/portrait"
    trans = set_transform_op()
    dataset = PortraitDataset(data_path, mode="train", transforms=trans)
    print(len(dataset))
    # print(dataset[1])
    for i in range(3):
        sample = dataset[i]
        img, label = sample["image"], sample["label"]
        print(img.dtype, label.dtype)
        image = dataset.get_image(i)
        plt.imshow(image, cmap='gray')
        plt.show()

    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    for i, batch in enumerate(data_loader):
        if i >= 3:
            break
        batch_img, batch_anno = batch["image"], batch["label"]
        print(batch_img.shape, batch_anno.shape)
