import glob
import os
from collections import defaultdict
import re
import numpy as np



class FoldDataCollect:

    def __init__(self, data_root="results/folds_ip"):
        self.data_root = data_root
        self.models = os.listdir(f"{self.data_root}/fold_0")
        self.log_files = collect_files(data_root)
        self.model_folds_log = self.separate_model_logs()
        self.model_folds_statistic = self.folds_data_statistic()
        self.print_model_info()
        print()
    
    def print_model_info(self):
        for model in self.models:
            mean = str(self.model_folds_statistic[model]["mean"])
            mean = mean.strip("[").strip("]").replace(",", " &")
            mean_ave = eval(mean.replace("&", "+")) / 5

            std = str(self.model_folds_statistic[model]["std"])
            std = std.strip("[").strip("]").replace(",", " &")
            std_ave = eval(std.replace("&", "+")) / 5

            print(model, "\n", "\t")
            print("\t", f"mean: {mean} & {np.round(mean_ave, 4)}")
            print("\t", f"std: {std} & {np.round(std_ave, 4)}")
    
    def folds_data_statistic(self):
        model_statistic = defaultdict(dict)
        for model in self.models:
            log_files = self.model_folds_log[model]
            mean, std = collect_log_file_data(log_files)
            model_statistic[model]["mean"] = mean.tolist()
            model_statistic[model]["std"] = std.tolist() 
        return model_statistic


    def separate_model_logs(self):
        model_logs = {}
        for model in self.models:
            model_logs[model] = [
                log for log in self.log_files if model in log
            ]
        return model_logs


def collect_log_file_data(log_files_list):
    records = []
    for log in log_files_list:
        with open(log, 'r') as fo:
            line = fo.readlines()[-1]
            assert "ccc" in line, "not correct data line"
            data = filter_data(line)
            records.append(data)
    assert len(records) == 10, "not complete collection"
    data_array = np.array(records)
    mean = data_array.mean(axis=0).round(4)
    std = data_array.std(axis=0).round(4)
    return mean, std


def filter_data(line):
    ptn = re.compile(r'[{}](.*?)[}]', re.S)
    data = re.findall(ptn, line)[0].split(",")
    assert len(data) == 5, "not enough traits"
    data_val = [float(it.split(":")[-1]) for it in data]
    return data_val


def collect_files(data_root):
    files = glob.glob(f"{data_root}/*/*/*/log.log")
    return files


if __name__ == "__main__":
    collector = FoldDataCollect()