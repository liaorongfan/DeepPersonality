import os
import os.path as opt
import glob
from pathlib import Path
from collections import defaultdict


def link_path(source: str, dest: str) -> None:
    """
    soft link data from source path to destination path
    
    Args:
        source(str): path to a source file
        dest(str): path of the destination file

    """
    if opt.exists(dest):
        dest = dest.rstrip("/")
        cmd = f"rm -rf {dest}"
        os.system(cmd)

    det_dir = opt.dirname(dest)
    os.makedirs(det_dir, exist_ok=True)

    cmd = f"ln -s {source} {dest}"
    assert os.system(cmd) == 0, \
        f"unable to link file: {source}"


def read_folding_file(file):
    with open(file, 'r') as f:
        fold_ls = [eval(line) for line in f.readlines()]
    return fold_ls


class LinkChalearn16Folding:

    def __init__(
        self, source_data_root="/home/rongfan/05-personality_traits/DeepPersonality/datasets",
        dest_data_root="./datasets/folds/fold_0",
        folding_file="datasets/folds_splits/10_fold_ip_video_id_ls.txt",
        mode="image",
        face_image=True,
    ) -> None:
        self.data_root = source_data_root
        self.dest_data_root = dest_data_root
        self.folding_file = folding_file
        self.fold_ls = read_folding_file(folding_file)
        self.all_video_frame, self.all_video_face, self.all_video_audio = self.collect()
        self.video_frame_dt, self.video_face_dt, self.video_audio_dt = self.indexing_dict()
        # self.mode = mode
        # self.face_image = face_image

    def collect(self):
        all_video_frame, all_video_face, all_video_audio = [], [], []
        for split in ["train", "valid", "test"]:

            frame_data_path = os.path.join(self.data_root, "image_data", f"{split}_data")
            video_frame = list(glob.glob(f"{frame_data_path}/*"))
            all_video_frame.extend(video_frame)

            face_data_path = os.path.join(self.data_root, "image_data", f"{split}_data_face")
            video_face = list(glob.glob(f"{face_data_path}/*"))
            all_video_face.extend(video_face)

            audio_data_path = os.path.join(self.data_root, "voice_data", "voice_librosa", f"{split}_data")
            video_audio = list(glob.glob(f"{audio_data_path}/*"))
            all_video_audio.extend(video_audio)

        return all_video_frame, all_video_face, all_video_audio

    def data_split(self, idx):
        assert idx < 10, "only support 10 folds"
        valid_idx = (idx + 1) % 10
        test_split = self.fold_ls[idx]
        valid_split = self.fold_ls[valid_idx]
        
        index = [i for i in range(10)]
        index.remove(idx)
        index.remove(valid_idx)
        train_split = []
        for i in index:
            train_split.extend(self.fold_ls[i])
        return train_split, valid_split, test_split

    def assemble_videos(self, train_split, valid_split, test_split, mode="frame"):
        if mode == "frame":
            data_dt = self.video_frame_dt
        elif mode == "face":
            data_dt = self.video_face_dt
        else:
            data_dt = self.video_audio_dt

        train_video_pt, valid_video_pt, test_video_pt = [], [], []
        for item in train_split:
            videos = data_dt[item]
            train_video_pt.extend(videos)
        for item in valid_split:
            videos = data_dt[item]
            valid_video_pt.extend(videos)
        for item in test_split:
            videos = data_dt[item]
            test_video_pt.extend(videos)

        return train_video_pt, valid_video_pt, test_video_pt

    def indexing_dict(self):
        frame_dt, face_dt, audio_dt = defaultdict(list), defaultdict(list), defaultdict(list)
        for video in self.all_video_frame:
            name, clip = os.path.basename(video).split(".")
            frame_dt[name].append(video)
        for video in self.all_video_face:
            name, clip = os.path.basename(video).split(".")
            face_dt[name].append(video)
        for video in self.all_video_audio:
            name, clip, _, _ = os.path.basename(video).split(".")
            audio_dt[name].append(video)
        return frame_dt, face_dt, audio_dt

    def setup_one_fold_dataset(self, idx: int):
        """
        Link source data and redistributed in different folds
        
        Args:
            idx(int): to select the idx-th rows in `fold_ls` as test fold
        
        """
        train_split, valid_split, test_split = self.data_split(idx)

        train_videos, valid_videos, test_videos = self.assemble_videos(
            train_split, valid_split, test_split, mode="frame",
        )
        self.redistribute_dataset(train_videos, valid_videos, test_videos, mode="frame")

        train_videos, valid_videos, test_videos = self.assemble_videos(
            train_split, valid_split, test_split, mode="face",
        )
        self.redistribute_dataset(train_videos, valid_videos, test_videos, mode="face")

        train_videos, valid_videos, test_videos = self.assemble_videos(
            train_split, valid_split, test_split, mode="audio",
        )
        self.redistribute_dataset(train_videos, valid_videos, test_videos, mode="audio")

    def redistribute_dataset(self, train_path_ls, valid_path_ls, test_path_ls, mode="frame"):
        splits = ["train", "valid", "test"]
        video_path_ls = [train_path_ls, valid_path_ls, test_path_ls]
        for video_path, split in zip(video_path_ls, splits):
            for video in video_path:
                if mode == "frame":
                    dest_path = f"{self.dest_data_root}/image_data/{split}_data"
                elif mode == "face":
                    dest_path = f"{self.dest_data_root}/image_data/{split}_data_face"
                elif mode == "audio":
                    dest_path = f"{self.dest_data_root}/voice_data/voice_librosa/{split}_data"
                else:
                    raise ValueError("mode must be specified")

                dest_path = os.path.join(dest_path, os.path.basename(video))
                link_path(video, dest_path)


if __name__ == "__main__":
    # link_path(
    #     "/hy-tmp/animal/animal_test/008105", 
    #     "/root/DeepPersonality/datasets/chalearn21_fold_1/xxx"
    # )
    link_folding = LinkChalearn16Folding(
        source_data_root="/root/DeepPersonality/datasets",
        dest_data_root="datasets/folds/fold_1",
    )
    link_folding.setup_one_fold_dataset(1)
