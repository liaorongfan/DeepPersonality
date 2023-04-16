import glob
import random
import collections
from collections import defaultdict
import itertools


class FoldingChalearn21Dataset:

    def __init__(
        self, 
        data_root: str = "/hy-tmp/ghost",
        valid_size: int = 18,
        k_fold: int = 10,
    ) -> None:
        self.data_root = data_root
        self.valid_size = valid_size
        self.dialog_video_id_ls = self.analysis_dataset()
        self.replacable_option = []
        self.test_folds, self.trainval_folds = self.split_folds()
        self.train_folds, self.valid_folds = self.split_trainval_folds()
        print("finish dataset folding")

    def analysis_dataset(self):

        dialog_video_id_ls = [
            video.split("/")[-1] 
            for video in glob.glob(f"{self.data_root}/*/[^0-9]*")
        ]
        dialog_video_id_ls, removed_video = self.dataset_rearrange(dialog_video_id_ls)     
        # _, _ = self.dataset_rearrange(dialog_video_id_ls)   
        return dialog_video_id_ls
        
    def dataset_rearrange(self, all_video_ids):
        sub_appear_five, sub_appear_four, sub_appear_three, \
        sub_appear_twice, sub_appear_once = self.statistic_dataset(all_video_ids)

        remove_5 = self.produce_subjects_show_5_times(
            all_video_ids, 
            sub_appear_five, sub_appear_four, 
            sub_appear_three, sub_appear_twice,
        )
        remove_4 = self.produce_subjects_show_4_times(
            all_video_ids, sub_appear_four, sub_appear_three, sub_appear_twice,
        )
        remove_3 = self.produce_subjects_show_3_times(
            all_video_ids, sub_appear_three, sub_appear_twice,
        )
        remove_2 = self.produce_subjects_show_2_times(
            all_video_ids, sub_appear_twice, sub_appear_once,
        )
        video_to_remove = remove_2 + remove_3 + remove_4 + remove_5

        video_to_remove = list(
            itertools.chain(
                *[
                    [item[0]+item[1], item[1] + item[0]]
                    for item in video_to_remove
                ]
            )
        )
        # clean video_to_remove list item in case some of them not a real video ID
        video_to_remove = list(set(all_video_ids) & set(video_to_remove))
        # remove frequently appeared subjects 
        new_dialog_video_ids = list(set(all_video_ids) - set(video_to_remove))

        return new_dialog_video_ids, video_to_remove

    def statistic_dataset(self, all_video_ids):
        sub_list = list(
            itertools.chain(
                *[
                    [sub[:3], sub[3:]] for sub in all_video_ids
                ]
            )
        )
        count_dic = collections.defaultdict(int)
        for sub in sub_list:
            count_dic[sub] += 1
        
        sub_appear_once = [k for k, v in count_dic.items() if v == 1]
        sub_appear_twice = [k for k, v in count_dic.items() if v == 2]
        sub_appear_three = [k for k, v in count_dic.items() if v == 3]
        sub_appear_four = [k for k, v in count_dic.items() if v == 4]
        sub_appear_five = [k for k, v in count_dic.items() if v == 5]

        return (
            sub_appear_five, sub_appear_four, sub_appear_three, 
            sub_appear_twice, sub_appear_once,
        )

    def find_subs(self, sub_ls, all_video_ids):
        # for sub in all_video_ids:
        sub_ls_info = {}
        for sub in sub_ls:
            sub_video_ls, sub_chain_ls = [], []
            for video in all_video_ids:
                if video[:3] == sub:
                    sub_video_ls.append(video)
                    sub_chain_ls.append(video[3:])
                elif video[3:] == sub:
                    sub_video_ls.append(video)
                    sub_chain_ls.append(video[:3])
        
            sub_ls_info[sub] = {
                "video": sub_video_ls, "related": sub_chain_ls
            }
        return sub_ls_info
            
    def produce_subjects_show_5_times(
        self, 
        all_video_ids, 
        sub_appear_five, sub_appear_four, 
        sub_appear_three, sub_appear_twice,
    ):
        video_to_remove = []
        # videos form which a subjects appeared 5 times 
        video_sub_5 = self.find_subs(sub_appear_five, all_video_ids)
        
        # subjects both appeared in 5 times and 4 times in the video datasets
        chain_5_to_4 = self.find_chain_sub(video_sub_5, sub_appear_four)

        for sub, relate in chain_5_to_4.items():
            for rel in relate:
                video_to_remove.append((sub, rel))
        
        # subjects both appeared in 5 times and 3 times in the video datasets
        chain_5_to_3 = self.find_chain_sub(video_sub_5, sub_appear_three)

        # those high frequent showing up subjects should be removed for folding
        for sub, relate in chain_5_to_3.items():
            for rel in relate:
                video_to_remove.append((sub, rel))
        # subjects both appeared in 5 times and 3 times in the video datasets
        chain_5_to_2 = self.find_chain_sub(video_sub_5, sub_appear_twice)
        for sub, relate in chain_5_to_2.items():
            for rel in relate:
                video_to_remove.append((sub, rel))
        # used to check whether all the subjects which show up 5 times
        remained_sub_5 = list(
            set(video_sub_5.keys()) - set([video[0] for video in video_to_remove])
        )

        if len(remained_sub_5) > 0:
            for sub in remained_sub_5:
                video_to_remove.append((sub, chain_5_to_2[sub][0]))  
        
        return video_to_remove

    def find_chain_sub(self, sub_info_dt, sub_ls):
        chain_info_dt = {}
        for k, v in sub_info_dt.items():
            chain_sub = list(set(v["related"]) & set(sub_ls))
            chain_info_dt[k] = chain_sub
        return chain_info_dt

    def produce_subjects_show_4_times(self,all_video_ids, sub_appear_four, sub_appear_three, sub_appear_twice):
        video_to_remove = []
        # videos form which a subjects appeared 4 times 
        video_sub_four = self.find_subs(sub_appear_four, all_video_ids)
        chain_4_to_3 = self.find_chain_sub(video_sub_four, sub_appear_three)
        # those high frequent showing up subjects should be removed for folding
        for sub, relate in chain_4_to_3.items():
            for rel in relate:
                video_to_remove.append((sub, rel))

        
        chain_4_to_2 = self.find_chain_sub(video_sub_four, sub_appear_twice)

        remained_sub_four = list(
            set(video_sub_four.keys()) - set([video[0] for video in video_to_remove])
        )
        if len(remained_sub_four) > 0:
            for sub in remained_sub_four:
                video_to_remove.append((sub, chain_4_to_2[sub][0])) 

        return video_to_remove

    def produce_subjects_show_3_times(self, all_video_ids, sub_appear_3, sub_appear_2):
        video_to_remove = []
        # videos form which a subjects appeared 3 times
        video_sub_3 = self.find_subs(sub_appear_3, all_video_ids)
        chain_3_to_2 = self.find_chain_sub(video_sub_3, sub_appear_2)
        specific_subs = {
            "182": chain_3_to_2["182"],
            "127": chain_3_to_2["127"],
            "139": chain_3_to_2["139"],
            "041": chain_3_to_2["041"],
            "005": chain_3_to_2["005"],
            "156": chain_3_to_2["156"],
        }
        # those high frequent showing up subjects should be removed for folding
        for sub, relate in specific_subs.items():
            for rel in relate[:1]:
                video_to_remove.append((sub, rel))

        return video_to_remove

    def produce_subjects_show_2_times(self, all_video_ids, sub_appear_2, sub_appear_1):
        video_to_remove = []
        # videos form which a subjects appeared 3 times
        video_sub_2 = self.find_subs(sub_appear_2, all_video_ids)
        chain_2_to_1 = self.find_chain_sub(video_sub_2, sub_appear_2)
        specific_subs = {
            "058": chain_2_to_1["058"],
            # "098": chain_2_to_1["098"],
        }
        # those high frequent showing up subjects should be removed for folding
        for sub, relate in specific_subs.items():
            for rel in relate[:1]:
                video_to_remove.append((sub, rel))

        return video_to_remove

    def separate_options(self, options, step=11):

        split_folds = []
        splits = list(range(0, len(options), step))
        for i in range(len(splits) - 1):
            fold_k_stat = splits[i]
            fold_k_end = splits[i + 1]
            split_folds.append(options[fold_k_stat: fold_k_end])
        return split_folds

    def split_folds(self):

        _, sub_appear_4, sub_appear_3, \
        sub_appear_2, sub_appear_1 = self.statistic_dataset(self.dialog_video_id_ls)
        dialog_video_id_ls = self.dialog_video_id_ls.copy()

        options_dt = {}

        options_3_dt, dialog_video_id_ls = self.select_video_by_show_up_times(
            dialog_video_id_ls, sub_appear_3,
        )

        options_2_dt, dialog_video_id_ls = self.select_video_by_show_up_times(
            dialog_video_id_ls, sub_appear_2,
        )

        options_1_dt, dialog_video_id_ls = self.select_video_by_show_up_times(
            dialog_video_id_ls, sub_appear_1,
        )

        all_video_ls = self.dialog_video_id_ls.copy()
        video_folds_finder = [{} for _ in range(10)]
        video_folds_lists = []

    def register_subjects(self, sub, all_dialog_videos=[], sub_info_dt=defaultdict(dict)):

        # sub_info_dt = collections.defaultdict(dict)
        sub_video_ls, sub_chain_ls = [], []

        for video in all_dialog_videos:
            if video[:3] == sub:
                sub_video_ls.append(video)
                sub_chain_ls.append(video[3:])
            elif video[3:] == sub:
                sub_video_ls.append(video)
                sub_chain_ls.append(video[:3])

        if sub not in sub_info_dt.keys():
            sub_info_dt[sub].update({"video": sub_video_ls, "chain": sub_chain_ls})
        else:
            print(f"{sub} has registered")
        # print()
        all_dialog_videos =  [video for video in all_dialog_videos if video not in sub_video_ls]

        for chain_sub in sub_chain_ls:
            sub_info_dt, all_dialog_videos = self.register_subjects(chain_sub, all_dialog_videos, sub_info_dt)

        return sub_info_dt, all_dialog_videos
    
    def select_video_by_show_up_times(self, dialog_video_id_ls, sub_appear_times):
        # options_info_dt = collections.defaultdict(dict)  # map sub: video relations
        sub_info_dt_ls = []
        for sub in sub_appear_times:
            # collect videos related to this subject
            sub_info_dt = defaultdict(dict)
            sub_info_dt, dialog_video_id_ls = self.register_subjects(sub, dialog_video_id_ls, sub_info_dt)
            sub_chain_ls = list(itertools.chain(*[v["video"] for k, v in sub_info_dt.items()]))
            sub_info_dt_ls.append(sub_chain_ls)

        # dialog_video_id_ls = list(set(dialog_video_id_ls) - set(options_ls))
        return sub_info_dt_ls, dialog_video_id_ls

    def split_train_folds(self):
        pass
    
    def find_duplicate(self, test_fold, trainval_fold):
        test_subs = itertools.chain(*[[sub[:3], sub[3:]] for sub in test_fold])
        trainval_subs = itertools.chain(*[[sub[:3], sub[3:]] for sub in trainval_fold])
        duplicate = [sub for sub in test_subs if sub in trainval_subs]
        return duplicate
        
    def varify_subject_independence(self, test_fold, trainval_fold):
        duplicate = self.find_duplicate(test_fold, trainval_fold)
        if len(duplicate) == 0:
            return test_fold, trainval_fold
        # elif len(duplicate) > 0:
        for item in duplicate:
            test_fold, trainval_fold = self.replace_subject(
                item, test_fold, trainval_fold,
            )

        return self.varify_subject_independence(test_fold, trainval_fold)                    
        
    def replace_subject(self, sub, test_fold, trainval_fold):
        # print(f"{sub}")
        remove = [
            video for video in test_fold 
            if (video[:3] == sub or video[3:] == sub)
        ]
        optional = [
            video for video in self.replacable_option 
            if not video in test_fold
        ]
        for it in remove:
            test_fold.remove(it)
            trainval_fold.append(it) 

            replaced_one = random.choice(optional)
            # self.replacable_option.remove(replaced_one)
            test_fold.append(replaced_one)
            trainval_fold.remove(replaced_one)
        
        return test_fold, trainval_fold

    def split_trainval_folds(self):
        train_folds, valid_folds = [], []
        for fold in self.trainval_folds:
            random.shuffle(fold)
            valid_folds.append(fold[: self.valid_size])
            train_folds.append(fold[self.valid_size: ])
        return train_folds, valid_folds

    def link_data(self):
        pass


if __name__ == "__main__":
    folder = FoldingChalearn21Dataset()
