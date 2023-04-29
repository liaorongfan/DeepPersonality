import glob
import os
from collections import defaultdict
import numpy as np


class FoldChalearn16:

    def __init__(self, data_root="./datasets"):
        self.data_root = data_root
        self.all_videos = self.collect()
        self.video_name_dt, self.len_matrix = self.indexing()
        self.fold_ls = self.folding()

    def collect(self):
        all_videos = []
        for split in ["train", "valid", "test"]:
            data_path = os.path.join(self.data_root, "image_data", f"{split}_data_face")
            videos = list(glob.glob(f"{data_path}/*"))
            all_videos.extend(videos)
        return all_videos

    def indexing(self):
        """
        remain video name for one person for example xxxx.001; xxxx.002
        will be selected as xxxx
        """
        video_name_dt = defaultdict(list)
        video_num_dt = defaultdict(int)
        for video in self.all_videos:
            name, clip = os.path.basename(video).split(".")
            video_name_dt[name].append(clip)
            video_num_dt[name] += 1

        max_len = max(video_num_dt.values())
        len_matrix = [[] for _ in range(6)]
        for i in range(6):
            l_th = i + 1
            tmp = [k for k, v in video_num_dt.items() if v == l_th]
            len_matrix[i].extend(tmp)

        # slice = [len(line) // 10 for line in len_matrix]
        # num_videos = 0
        # for i, num in enumerate(slice):
        #     print(i)
        #     num_videos += num * (i + 1)

        return video_name_dt, len_matrix

    def folding(self):
        slices = [len(line) // 10 for line in self.len_matrix]
        len_matrix = [[] for _ in range(6)]
        for i in range(6):
            line = self.len_matrix[i]
            step = slices[i]
            len_matrix[i] = [line[ii: ii + step] for ii in range(0, len(line), step)]
        align_matrix = [line[:10] for line in len_matrix]
        extra_matrix = [line[10:] for line in len_matrix]
        folding_matrix = [[] for _ in range(10)]
        for i in range(10):
            for ii in range(6):
                folding_matrix[i].extend(align_matrix[ii][i])

        for i, item in enumerate(extra_matrix[5][0]):
            folding_matrix[i].append(item)
        extra_matrix[5] = []
        folding_matrix[9].append(extra_matrix[2][0].pop())
        folding_matrix[9].append(extra_matrix[2][0].pop())

        for i, item in enumerate(extra_matrix[3][0]):
            folding_matrix[i].append(item)
        extra_matrix[3] = []
        folding_matrix[8].append(extra_matrix[1][0].pop())
        folding_matrix[8].append(extra_matrix[1][0].pop())
        folding_matrix[9].append(extra_matrix[2][0].pop())

        # with open("ip_fold_ls.txt", 'w') as f:
        #     for line in folding_matrix:
        #         f.writelines(line, "\n")
        return folding_matrix

    def write_folding_file(self):
        with open("ip_fold_ls.txt", 'w') as f:
            for line in self.fold_ls:
                f.writelines(f"{line}\n")


if __name__ == '__main__':
    folder = FoldChalearn16()
    folder.write_folding_file()
