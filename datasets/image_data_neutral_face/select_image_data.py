import os
import pandas
import glob


# iter through all csv in the folder
def iter_all_txt(csv_dir_path, src_img_root="../image_data/test_data_face", dst_img_root="./test_data_face"):
    au_files = glob.glob(f"{csv_dir_path}/*/*.csv")
    for au_file in au_files:
        video_name = os.path.basename(au_file).replace(".csv", "")

        src_img_dir = os.path.join(src_img_root, video_name)

        img_idx = parse_csv(au_file)
        # print(img_idx)
        for idx in img_idx:
            src_img_file = os.path.join(src_img_dir, f"face_{idx}.jpg")

            os.makedirs(os.path.join(dst_img_root, video_name), exist_ok=True)
            det_img_file = os.path.join(dst_img_root, video_name, f"face_{idx}.jpg")

            print(src_img_file, " ---> ",  det_img_file)
            os.system(f"cp {src_img_file} {det_img_file}")


def parse_csv(csv_path, select_num=5):
    csv = pandas.read_csv(csv_path)
    one = csv[" AU45_c"] < 1
    two = csv[" AU28_c"] < 1
    three = csv[" AU26_c"] < 1
    four = csv[" AU25_c"] < 1
    five = csv[" AU23_c"] < 1
    fix = csv[" AU20_c"] < 1
    select = csv[one & two & three & four & five & fix]["frame"].tolist()
    if len(select) < 5:
        select = csv[one & two & three & four & five]["frame"].tolist()
    return select[:select_num]


if __name__ == '__main__':
    iter_all_txt(
        csv_dir_path="./train_data_face_aus",
        src_img_root="../image_data/train_data_face",
        dst_img_root="./train_data_face",
    )
