import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FlowerDataset(Dataset):
    cls_num = 102
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir,  mode="train", transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.data_root = root_dir
        self.img_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.img_info = []   # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img

    def __len__(self):
        """
        返回数据集的长度
        :return:
        """
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        names_imgs = os.listdir(self.img_dir)
        names_imgs = [n for n in names_imgs if n.endswith(".jpg")]
        if len(names_imgs) == 0:
            raise Exception("\ndata_dir:{} has no images! Please checkout your path to images!".format(
                self.img_dir))
        idx_imgs = [int(n[6:11]) for n in names_imgs]
        path_imgs = [os.path.join(self.img_dir, n) for n in names_imgs]

        # 读取mat形式label
        label_file = "imagelabels.mat"  # hard code
        path_label_file = os.path.join(self.data_root, label_file)
        from scipy.io import loadmat
        self.label_array = loadmat(path_label_file)["labels"].squeeze()

        # 匹配label # 注意索引，注意标签减一
        self.img_info = [(p, int(self.label_array[idx - 1] - 1)) for p, idx in zip(path_imgs, idx_imgs)]


def make_data_loader(cfg, mode):
    if mode == "train":
        transforms = cfg.transforms_train
    elif mode == "valid":
        transforms = cfg.transforms_valid
    elif mode == "test":
        transforms = cfg.transforms_valid
    else:
        raise ValueError("mode only supports 'train', 'valid' and 'test'")

    dataset = FlowerDataset(cfg.flower_data_root, mode, transforms)
    data_loader = DataLoader(dataset=dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    return data_loader


if __name__ == "__main__":

    # root_dir = r"G:\deep_learning_data\102flowers\train"
    root_dir = r"F:\23-deepshare\09-deep_share_cv_code\datasets\102flowers"

    test_dataset = FlowerDataset(root_dir)

    print(len(test_dataset))
    print(next(iter(test_dataset)))


