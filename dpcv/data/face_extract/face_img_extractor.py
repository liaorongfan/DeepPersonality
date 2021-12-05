import os
import cv2
from tqdm import tqdm
import glob
from data.face_extract.face_detector import FaceDetection


class FaceImageExtractor:

    def __init__(self, data_root, detector_path):
        """
        args:
            data_root:(str) : where to save the processed video directory
            detector_path:(str) : the path to dlib face detector model
        """
        self.data_root = data_root
        self.video_name = None
        self.video_file = None
        self.save_dir = None
        self.frame_count = 0
        self.frame_count_new = 0
        self.fps = 0
        self.duration = 0
        self.step = 0

        self.face_detector = FaceDetection(detector_path + "shape_predictor_68_face_landmarks.dat")

    def load_video(self, video_name):
        """extract face images form a video and the video name will be the name of a directory to save face images

        args:
            video_name:(str) name of video file end with ".mp4"
        """
        self.video_name = os.path.basename(video_name)[:-4]

        self.save_dir = os.path.join(self.data_root, self.video_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.video_file = cv2.VideoCapture(video_name)

    def reduce_frame_rate(self, fps_new):
        """ extract face images with the sample rate of fps_new

        args:
            fps_new:(int) face images sample rate
        return:
            save detected face images to specified directory
        """
        ret = self.video_file.isOpened()
        if ret:
            self.frame_count = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = int(self.video_file.get(cv2.CAP_PROP_FPS))
            self.duration = int(self.frame_count / self.fps)

            self.frame_count_new = self.duration * fps_new
            self.step = int(self.frame_count / self.frame_count_new)

            print("----------------------------")
            print("frame_count original: " + str(self.frame_count))
            print("fps original: " + str(self.fps))
            print("duration: " + str(self.duration))
            print("----------------------------")
            print("frame_count new: " + str(self.frame_count_new))
            print("fps new: " + str(fps_new))
            print("duration: " + str(self.duration))
            print("step: " + str(self.step))
            print("----------------------------")

            if self.step == 0:
                self.step = 1
                print("WARNING: selected framerate is higher as original!")

            cnt = 0
            for i in tqdm(range(0, int(self.frame_count), int(self.step))):
                cnt = cnt + 1
                ret, frame = self.video_file.read()
                if ret:
                    if self.face_detector.find_face(frame):
                        frame_crop = self.face_detector.run(frame)
                        cv2.imwrite(f"{self.save_dir}/face_{cnt}.jpg", frame_crop)

                if cnt >= self.frame_count_new:
                    break

    def process_frames(self):
        ret = self.video_file.isOpened()
        if ret:
            frame_count = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT)

            cnt = 0
            for i in tqdm(range(int(frame_count))):
                cnt = cnt + 1
                ret, frame = self.video_file.read()
                if ret:
                    if self.face_detector.find_face(frame):
                        frame_crop = self.face_detector.run(frame)
                        cv2.imwrite(f"{self.save_dir}/face_{cnt}.jpg", frame_crop)

                if cnt >= frame_count:
                    break

    def play_video(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.video_file.read()
            if ret:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        self.video_file.release()
        cv2.destroyAllWindows()


def run_on_videos(video_dir):
    image_extractor = FaceImageExtractor(
        data_root="/home/rongfan/11-personality_traits/DeepPersonality/datasets/image_data/train_data_face_66-70",
        detector_path="/home/rongfan/11-personality_traits/DeepPersonality/checkpoints/"
    )
    input_video_ls = glob.glob(f"{video_dir}/*.mp4")
    # input_video = "/home/rongfan/11-personality_traits/apa_paper/FaceDBGenerator_V2/Facedetector/_QXI4n_FRN4.003.mp4"
    for input_video in input_video_ls:
        image_extractor.load_video(input_video)
        # image_extractor.reduce_frame_rate(4)
        image_extractor.process_frames()


if __name__ == "__main__":
    vid_dirs = [
        "/home/rongfan/11-personality_traits/DeepPersonality/datasets/unzipped_data/training80_66",
        "/home/rongfan/11-personality_traits/DeepPersonality/datasets/unzipped_data/training80_67",
        "/home/rongfan/11-personality_traits/DeepPersonality/datasets/unzipped_data/training80_68",
        "/home/rongfan/11-personality_traits/DeepPersonality/datasets/unzipped_data/training80_69",
        "/home/rongfan/11-personality_traits/DeepPersonality/datasets/unzipped_data/training80_70",
    ]
    for dir in vid_dirs:
        run_on_videos(dir)
