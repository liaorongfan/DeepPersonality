import glob
import os
from collections import defaultdict
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


class AUDataCollect:

    TRAITS = ["Ope", "Con", "Ext", "Agr", "Neu"]

    def __init__(self, data_root=None):
        self.ip_au_log_files = collect_au_log_files("results_second_stage/action_units/ip")
        self.tp_au_log_files = collect_au_log_files("results_second_stage/action_units/tp")
        self.au_ls = self.get_au_ls()
        self.ip_data_array = self.assemble_log_data(self.ip_au_log_files)
        self.tp_data_array = self.assemble_log_data(self.tp_au_log_files)

    @staticmethod
    def assemble_log_data(log_file_ls):
        records = []
        for log in log_file_ls:
            with open(log, 'r') as fo:
                line = fo.readlines()[-1]
                assert "ccc" in line, "not correct data line"
                data = np.array(filter_data(line)).round(4)
                records.append(data)
        assert len(records) == 17, "not complete collection"
        data_array = np.array(records)
        return data_array

    def get_au_ls(self):
        name_ls = []
        for log in self.ip_au_log_files:
            *_, au, data, file = log.split("/")
            name_ls.append(au[:4])
        return name_ls

    def get_traits_au(self, trait):
        if trait == "M":
            aus_ip = self.ip_data_array.mean(axis=1) * 100  # .tolist()
            aus_tp = self.tp_data_array.mean(axis=1) * 100  # .tolist()
            return aus_ip, aus_tp

        if trait == "T":
            aus_ip = self.ip_data_array.mean(axis=0) * 100  # .tolist()
            aus_tp = self.tp_data_array.mean(axis=0) * 100  # .tolist()
            return aus_ip, aus_tp


        idx = ["O", "C", "E", "A", "N"].index(trait)
        aus_ip = self.ip_data_array[:, idx] * 100  # .tolist()
        aus_tp = self.tp_data_array[:, idx] * 100  # .tolist()
        return aus_ip, aus_tp


def collect_au_log_files(data_root):
    logs = sorted(glob.glob(f"{data_root}/*/*/log.log"))
    return list(logs)


class CompareGraph:
    def __init__(self, labels, input_1, input_2, x_label):
        x = np.arange(len(labels)) * 4  # the label locations
        width = 1.2  # the width of the bars

        self.fig, self.ax = plt.subplots()
        self.frame = self.ax.bar(x - width / 2, input_1, width, label='Apparent personality')
        self.face = self.ax.bar(x + width / 2, input_2, width, label='True personality')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        self.ax.set_ylabel('CCC (%)', fontsize=15)
        # ax.set_title('Impression ACC scores by frame and face images')
        traits_label = {
            "O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism",
            "M": "OCEAN Average From 17 Action Units", "T": "AUs Average on OCEAN traits"
        }[x_label]
        self.ax.set_xlabel(traits_label, fontsize=15)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.legend()

    def autolabel_frame(self, rects, xytxt=(-2, 2)):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            self.ax.annotate('{:.2f}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=xytxt,  # 3 points vertical offset
                             textcoords="offset points",
                             fontsize=6,
                             ha='center', va='bottom')

    def autolabel_face(self, rects, xytxt=(4, 2)):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            value = height
            if height < 0:
                height = height - 1.2
            self.ax.annotate('{:.2f}'.format(value),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=xytxt,  # 3 points vertical offset
                             textcoords="offset points",
                             fontsize=6,
                             ha='center', va='bottom')

    def draw(self, name):
        self.autolabel_frame(self.frame)
        self.autolabel_face(self.face)

        self.fig.tight_layout()
        # plt.figure(figsize=(12, 12))
        plt.ylim(-8, 25)
        # plt.xlim(-0.4, 30)
        plt.xticks(rotation=-30, size=8)
        # plt.xticks(np.linspace(0, 40, 17))
        plt.savefig(name, dpi=800)
        plt.show()


if __name__ == "__main__":
    collector = AUDataCollect()
    # collector = FoldDataCollect()
    labels = collector.au_ls
    # labels = AUDataCollect.TRAITS
    labels.append("Mean")
    # for t in ["O", "C", "E", "A", "N"]:
    #     aus_ip, aus_tp = collector.get_traits_au(t)
    #     graph = CompareGraph(labels, aus_ip, aus_tp, x_label=t)
    #     graph.draw()
    img_name = "A_17AUs.png"
    t = "A"
    aus_ip, aus_tp = collector.get_traits_au(t)
    aus_ip_mean = aus_ip.mean().astype(np.float32).round(4)[None]
    aus_tp_mean = aus_tp.mean().astype(np.float32).round(4)[None]
    # aus_ip, aus_tp = aus_ip.tolist(), aus_tp.tolist()
    # aus_ip = aus_ip.append(aus_ip_mean)
    # aus_tp = aus_tp.append(aus_tp_mean)
    aus_ip_m = np.concatenate([aus_ip, aus_ip_mean], axis=0)
    aus_tp_m = np.concatenate([aus_tp, aus_tp_mean], axis=0)
    graph = CompareGraph(labels, aus_ip_m, aus_tp_m, x_label=t)
    graph.draw(img_name)


