"""
code modified form https://github.com/miguelmore/personality
"""
import os
import numpy as np
import pickle


class DataSet:

    # 30935 total
    total_train = 28152
    total_test = 2783

    def __init__(self, data_path, mode="train"):
        self.data_path = data_path
        # self.mode = mode
        self.data_set = self._load_dataset(data_path, mode)

    def _load_dataset(self, mode):
        if mode == "train":
            data_train = {'X': [], 'Y': []}
            for i in range(1, 4):
                x, y = self.read_pickle('train_clselfie_v4_{}.pickle'.format(i), show=True)
                data_train['X'].extend(x)
                data_train['Y'].extend(y)
            data_train['X'] = np.array(data_train['X'])
            data_train['Y'] = np.array(data_train['Y'])
            print('\nTotal Train Data X:', data_train['X'].shape, 'Y:', data_train['Y'].shape)
            return data_train
        elif mode == "test":
            data_test = {'X': [], 'Y': []}
            x, y = self.read_pickle('test_clselfie_v4.pickle', show=True)
            data_test['X'].extend(x)
            data_test['Y'].extend(y)

            data_test['X'] = np.array(data_test['X'])
            data_test['Y'] = np.array(data_test['Y'])
            print('Total Test  Data X:', data_test['X'].shape, 'Y:', data_test['Y'].shape)
            return data_test
        else:
            raise ValueError("data loading only support mode of 'train' or 'test' ")

    def read_pickle(self, name, show=False):
        path = os.path.join(self.data_path, name)
        pic = pickle.load(open(path, "rb"))
        x = np.array(pic['X'])
        y = np.array(pic['Y'])
        if show:
            print('X-', x.shape)
            print('Y-', y.shape)
        return x, y

    def normalize_images(self):
        train_x = []
        test_x = []
        for i in range(self.data_train['X'].shape[0]):
            train_x.append(self.data_train['X'][i] / 255)

        for i in range(self.data_test['X'].shape[0]):
            test_x.append(self.data_test['X'][i] / 255)

        self.data_train['X'] = np.array(train_x)
        self.data_test['X'] = np.array(test_x)

        print('Total Train Data X:', self.data_train['X'].shape, 'Y:', self.data_train['Y'].shape)
        print('Total Test  Data X:', self.data_test['X'].shape, 'Y:', self.data_test['Y'].shape)


root_dir = r"G:\deep_learning_data\102flowers\train"
