import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def data_clean(dir, save_dir):
    dir_ls = os.listdir(dir)
    for dir_i in tqdm(dir_ls):
        dir_path = os.path.join(dir, dir_i)
        img_path_ls = glob.glob(f"{dir_path}/*.bmp")

        img_saved_dir = os.path.join(save_dir, dir_i)
        if not os.path.exists(img_saved_dir):
            os.mkdir(img_saved_dir)

        for img_path in img_path_ls:
            img_name = os.path.basename(img_path).replace("bmp", "jpg")
            img_arr = Image.open(img_path).convert("RGB")
            img_arr_np = np.array(img_arr)
            if img_arr_np.sum() > 0:

                img_saved = os.path.join(img_saved_dir, img_name)
                img_arr.save(img_saved)


def img_dir_filter(data_dir):
    dir_ls = os.listdir(data_dir)
    for img_dir in dir_ls:
        img_path_ls = glob.glob(f"{data_dir}/{img_dir}/*.jpg")
        if len(img_path_ls) == 0:
            print(img_dir)


if __name__ == "__main__":
    # data_clean("trainingData_face/10", "face_train")
    img_dir_filter("face_valid")
