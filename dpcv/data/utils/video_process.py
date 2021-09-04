from data.utils.video_to_image import frame_sample
import os


def process():
    test_split = os.listdir("./test")
    for dir in test_split:
        dir_name = f"./test/{dir}/{dir}"
        video_list = os.listdir(dir_name)
        for video in video_list:
            video_path = os.path.join(dir_name, video)
            frame_sample(video_path, "../../../datasets/ImageData/")


if __name__ == "__main__":
    print(os.getcwd())
    process()
