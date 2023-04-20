import os
import os.path as opt
import glob


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


class LinkFolding:

    def __init__(self,source_data_root, dest_data_root, folding_file) -> None:
        self.data_root = source_data_root
        self.dest_data_root = dest_data_root
        self.folding_file = folding_file
        self.fold_ls = read_folding_file(folding_file)
        self.animal_data = self.assemble_data(session="animal")
        self.lego_data = self.assemble_data(session="lego")
        self.talk_data = self.assemble_data(session="talk")
        self.ghost_data = self.assemble_data(session="ghost")

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
        return test_split, valid_split, train_split

    def assemble_data(self, session: str) -> dict:
        assert session in ["animal", "lego", "talk", "ghost"], \
            "session should be one of [animal, lego, talk, ghost]"
        
        all_videos = glob.glob(f"{self.data_root}/{session}/*/*")
        all_videos_dt = {opt.basename(video): video for video in all_videos}
        return all_videos_dt
    
    def setup_one_fold_dataset(self, idx: int):
        """
        Link source data and redistributed in different folds
        
        Args:
            idx(int): to select the idx-th rows in `fold_ls` as test fold
        
        """
        test_split, valid_split, train_split = self.data_split(idx)
        self.redistribute_dataset("test", test_split)
        self.redistribute_dataset("valid", valid_split)
        self.redistribute_dataset("train", train_split)
    
    def redistribute_dataset(self, mode: str, split: list):
        for video_id in split:
            source_video_path = [
                self.animal_data[video_id],
                self.lego_data[video_id],
                self.talk_data[video_id],
                self.ghost_data[video_id],
            ]
            dest_path = [
                opt.join(self.dest_data_root, mode, f"{session}_{mode}", video_id)
                for session in ["animal", "lego", "talk", "ghost"]                
            ]
            for source, dest in zip(source_video_path, dest_path):
                # print(f"{source: <}\n{dest: >}\n")
                link_path(source, dest)


if __name__ == "__main__":
    # link_path(
    #     "/hy-tmp/animal/animal_test/008105", 
    #     "/root/DeepPersonality/datasets/chalearn21_fold_1/xxx"
    # )
    link_folding = LinkFolding(
        source_data_root="/hy-tmp",
        dest_data_root="/root/DeepPersonality/datasets/chalearn21_fold_7",
        folding_file="config/folds/10_fold_video_id_ls.txt",
    )
    link_folding.setup_one_fold_dataset(7)