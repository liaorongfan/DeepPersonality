import glob
import scipy.stats as stats
import os
from collections import defaultdict
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

class FoldDataCollect:

    def __init__(self, data_root="results/folds_tp", print_info=False):
        self.data_root = data_root
        self.models = os.listdir(f"{self.data_root}/fold_0")
        self.log_files = collect_files(data_root)
        self.model_folds_log = self.separate_model_logs()
        self.model_folds_statistic = self.folds_data_statistic()
        if print_info:
            self.print_model_info()

    def print_model_info(self):
        for model in self.models:
            mean = str(self.model_folds_statistic[model]["mean"])
            mean = mean.strip("[").strip("]").replace(",", " &")

            std = str(self.model_folds_statistic[model]["std"])
            std = std.strip("[").strip("]").replace(",", " &")

            print(model, "\n", "\t")
            print("\t", f"mean: {mean}")
            print("\t", f"std: {std} ")

    def folds_data_statistic(self):
        model_statistic = defaultdict(dict)
        for model in self.models:
            log_files = self.model_folds_log[model]
            data_array, mean, std = collect_log_file_data(log_files)
            model_statistic[model]["data"] = data_array  # .tolist()
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

    def compute_p_value(self, base_model):
        # remove base model from models list
        models = deepcopy(self.models)
        models.remove(base_model)
        # compute p value for each trait and each model
        for model in models:
            p_value_list = []
            # for each trait
            for trait in range(6):
                # compute p value
                p_value = self.p_value(base_model, model, trait)
                # round p value
                p_value = np.round(p_value, 6)
                p_value_str = f"{p_value:.6f}"
                p_value_list.append(p_value_str)  # f"{p_value: .6f}"
            # clean data for latex table
            p_value_list = clean_data(str(p_value_list))
            print(f"{base_model} vs {model}: {p_value_list}")
            print()

    def p_value(self, model_1, model_2, trait):
        # get data from each model by name and trait
        data_1 = self.model_folds_statistic[model_1]["data"][:, trait]
        data_2 = self.model_folds_statistic[model_2]["data"][:, trait]
        # compute p value
        t_statistic, p_value = stats.ttest_ind(data_1, data_2)
        return p_value


def clean_data(data):
    data = data.strip("[").strip("]").replace(",", " &").replace("'", "")
    return data


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
    trait_mean = data_array.mean(axis=1).round(4).reshape(10, 1)
    data_array_exp = np.hstack((data_array, trait_mean))
    mean = data_array_exp.mean(axis=0).round(4)
    # the mean of std of each trait
    std = data_array.std(axis=0).round(4)
    std_exp = np.hstack((std, std.mean().round(4)))
    return data_array_exp, mean, std_exp


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
    # do not use scientific notation
    np.set_printoptions(suppress=True)

    collector = FoldDataCollect()
    collector.compute_p_value("02_hrnet")



