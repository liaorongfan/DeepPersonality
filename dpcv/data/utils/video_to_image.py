import cv2
import numpy as np
import os
import zipfile
from tqdm import tqdm


def process():
    data_dir = "./chalearn/test"
    video_list = os.listdir(data_dir)
    for video in tqdm(video_list):
        video_path = os.path.join(data_dir, video)
        frame_sample(video_path, "./ImageData/testData/")


def frame_sample(video, save_dir):
    """
    Creating folder to save all the 100 frames from the video
    """
    cap = cv2.VideoCapture(video)

    file_name = (os.path.basename(video).split('.mp4'))[0]
    try:
        if not os.path.exists(save_dir + file_name):
            os.makedirs(save_dir + file_name)
    except OSError:
        print('Error: Creating directory of data')

    # Setting the frame limit to 100
    # cap.set(cv2.CAP_PROP_FRAME_COUNT, 120)
    # print(cap.get(cv2.CAP_PROP_FPS))
    # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5))
    cap.set(cv2.CAP_PROP_FPS, 25)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5)
    count = 0
    # Running a loop to each frame and saving it in the created folder
    while cap.isOpened():
        count += 1
        if length == count:
            break
        ret, frame = cap.read()
        if frame is None:
            continue
        # Resizing it to 256*256 to save the disk space and fit into the model
        frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame in jpg file
        name = save_dir + str(file_name) + '/frame' + str(count) + '.jpg'
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # print(f"{video} precessed")


def video2img_train(zipfile_train, save_to="image_data/train_data"):
    # Running a loop through all the zipped training file to extract all video and then extract 100 frames from each.
    for i in tqdm(range(1, 76)):
        if i < 10:
            zipfilename = 'training80_0' + str(i) + '.zip'
        else:
            zipfilename = 'training80_' + str(i) + '.zip'
        # Accessing the zipfile i
        archive = zipfile.ZipFile(zipfile_train + zipfilename, 'r')
        zipfilename = zipfilename.split('.zip')[0]

        # Extracting all videos in it and saving it all to the new folder with same name as zipped one
        archive.extractall('unzippedData/' + zipfilename)

        # Running a loop over all the videos in the zipped file and extracting 100 frames from each
        for file_name in tqdm(archive.namelist()):
            cap = cv2.VideoCapture('unzippedData/' + zipfilename + '/' + file_name)

            file_name = (file_name.split('.mp4'))[0]
            save_path = os.path.join(save_to, file_name)
            # Creating folder to save all the 100 frames from the video
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            except OSError:
                print('Error: Creating directory of data')

            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            count = 0
            # Running a loop to each frame and saving it in the created folder
            while cap.isOpened():
                count += 1
                if length == count:
                    break
                ret, frame = cap.read()
                if frame is None:
                    continue
                # Resizing it to 456*256 to save the disk space and fit into the model
                frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
                # Saves image of the current frame in jpg file
                name = f"{save_path}/frame_{str(count)}.jpg"
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def video2img_val(zipfile_val, saved_to="image_data/valid_data"):
    for i in tqdm(range(1, 26)):
        if i < 10:
            zipfilename = 'validation80_0' + str(i) + '.zip'
        else:
            zipfilename = 'validation80_' + str(i) + '.zip'
        # Accessing the zipfile i
        archive = zipfile.ZipFile(os.path.join(zipfile_val + zipfilename), 'r')
        zipfilename = zipfilename.split('.zip')[0]

        # Extracting all videos in it and saving it all to the new folder with same name as zipped one
        archive.extractall('unzipped_data/' + zipfilename)

        # Running a loop over all the videos in the zipped file and extracting 100 frames from each
        for file_name in tqdm(archive.namelist()):
            cap = cv2.VideoCapture('unzipped_data/' + zipfilename + '/' + file_name)

            file_name = file_name.split('.mp4')[0]
            # Creating folder to save all the 100 frames from the video
            save_path = os.path.join(saved_to, file_name)
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            except OSError:
                print('Error: Creating directory of data')

            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            count = 0
            # Running a loop to each frame and saving it in the created folder
            while cap.isOpened():
                count += 1
                if length == count:
                    break
                ret, frame = cap.read()
                if frame is None:
                    continue

                # Resizing it to (w, h) = (456, 256) to save the disk space and fit into the model
                frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
                # Saves image of the current frame in jpg file
                name = f"{save_path}/frame_{str(count)}.jpg"
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def video2img_test(zipfile_test, saved_to="image_data/test_data"):
    for i in range(7, 11):
        zipfilename = 'test_' + str(i) + '.zip'
        # Accessing the zipfile i
        archive = zipfile.ZipFile(os.path.join(zipfile_test + zipfilename), 'r')
        zipfilename = zipfilename.split('.zip')[0]

        # Extracting all videos in it and saving it all to the new folder with same name as zipped one
        archive.extractall('unzipped_data/' + zipfilename)

        # Running a loop over all the videos in the zipped file and extracting 100 frames from each
        for file_name in tqdm(archive.namelist()):
            cap = cv2.VideoCapture('unzipped_data/' + zipfilename + '/' + file_name)

            file_name = file_name.split('.mp4')[0]
            # Creating folder to save all the 100 frames from the video
            save_path = os.path.join(saved_to, file_name)
            try:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            except OSError:
                print('Error: Creating directory of data')

            # Setting the frame limit to 100
            # cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
            # length = 101
            # cap.set(cv2.CAP_PROP_FPS, 25)
            # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5)

            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if length == 0:
                print("escape:", file_name)
                continue
            count = 0
            # Running a loop to each frame and saving it in the created folder
            while cap.isOpened():
                count += 1
                if length == count:
                    break
                ret, frame = cap.read()
                if frame is None:
                    continue

                # Resizing it to (w, h) = (456, 256) to save the disk space and fit into the model
                frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
                # Saves image of the current frame in jpg file
                name = f"{save_path}/frame_{str(count)}.jpg"
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    # video2img_train("./chalearn/train/")
    # print("current work directory:", os.getcwd())
    # video2img_val("/home/rongfan/11-personality_traits/DeepPersonality/datasets/raw_data_tmp/validate/")
    # video2img_test("/home/rongfan/11-personality_traits/DeepPersonality/datasets/raw_data_tmp/test/")
    video2img_train(
        "/home/ssd500/personality_data/raw_data_tmp/train/", "/home/ssd500/personality_data/image_data/train_data")
