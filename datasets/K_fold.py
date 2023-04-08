import glob
import random
import collections
import itertools


class FoldingChalearn21Dataset:

    def __init__(
        self, 
        data_root: str = "/hy-tmp/ghost",
        valid_size: int = 18,
    ) -> None:
        self.data_root = data_root
        self.valid_size = valid_size
        self.dialog_video_id_ls = [
            video.split("/")[-1] 
            for video in glob.glob(f"{self.data_root}/*/[^0-9]*")
        ]
        self.test_folds, self.trainval_folds = self.split_folds()
        self.train_folds, self.valid_folds = self.split_trainval_folds()

    def split_folds(self):

        sub_list = list(
            itertools.chain(
                *[
                    [sub[:3], sub[3:]] for sub in self.dialog_video_id_ls
                ]
            )
        )
        count_dic = collections.defaultdict(int)
        for sub in sub_list:
            count_dic[sub] += 1
        
        sub_appear_once = [k for k, v in count_dic.items() if v == 1]
        sub_appear_twice = [k for k, v in count_dic.items() if v == 2]
        sub_appear_multi = [k for k, v in count_dic.items() if v >= 3]

        uni_optional = []
        for video in self.dialog_video_id_ls:
            sub1, sub2 = video[:3], video[3:]
            if (sub1 in sub_appear_once) and (sub2 not in sub_appear_multi):
                uni_optional.append(video)
            elif (sub2 in sub_appear_once) and (sub1 not in sub_appear_multi):
                uni_optional.append(video)
           
        self.replacable_option = uni_optional

        bi_optional = []
        for sub in sub_appear_twice:
            for video in self.dialog_video_id_ls:
                sub1, sub2 = video[:3], video[3:]
                if (sub1 == sub) and (sub2 not in sub_appear_multi):
                    bi_optional.append(video)
                elif (sub2 == sub) and (sub1 not in sub_appear_multi):
                    bi_optional.append(video)
        bi_optional = list(set(bi_optional))

        test_split_folds = []
        splits = list(range(0, len(bi_optional), 11))
        for i in range(len(splits) - 1):
            fold_k_stat = splits[i]
            fold_k_end = splits[i + 1]
            test_split_folds.append(bi_optional[fold_k_stat: fold_k_end])

        varified_test_folds, varified_trainval_folds = [], []
        for test_fold in test_split_folds:
            trainval_fold = [
                video for video in self.dialog_video_id_ls 
                if video not in test_fold
            ]
            test_fold, trainval_fold = self.varify_subject_independence(
                test_fold, trainval_fold
            )
            varified_test_folds.append(test_fold)
            varified_trainval_folds.append(trainval_fold)

        return varified_test_folds, varified_trainval_folds

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
            self.replacable_option.remove(replaced_one)
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
