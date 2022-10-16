import os
import torch
from tqdm import tqdm
import glob
import numpy as np
from scipy import signal
import pickle


def gen_statistic_data(data_path, save_to, method=None):
    data = torch.load(data_path)
    statistic_data_ls = []
    for sample in data["video_frames_pred"]:
        statistic_data_ls.append(method(sample))
    statistic_data = {"video_statistic": statistic_data_ls, "video_label": data["video_label"]}
    torch.save(statistic_data, save_to)
    print()


def assemble_pred_statistic(data):
    assert isinstance(data, torch.Tensor), "the input data should be torch.Tensor"
    max_0, _ = data.max(dim=0)
    min_0, _ = data.min(dim=0)
    mean_0 = data.mean(dim=0)
    std_0 = data.std(dim=0)

    data_first_order = data[1:, :] - data[:-1, :]
    max_1, _ = data_first_order.max(dim=0)
    min_1, _ = data_first_order.min(dim=0)
    mean_1 = data_first_order.mean(dim=0)
    std_1 = data_first_order.std(dim=0)

    data_sec_order = data[2:, :] - data[:-2, :]
    max_2, _ = data_sec_order.max(dim=0)
    min_2, _ = data_sec_order.min(dim=0)
    mean_2 = data_sec_order.mean(dim=0)
    std_2 = data_sec_order.std(dim=0)

    statistic_representation = torch.stack(
        [max_0, min_0, mean_0, std_0, max_1, min_1, mean_1, std_1, max_2, min_2, mean_2, std_2],
        dim=0,
    )

    return statistic_representation


def resample_pred_spectrum(data, new_rate=100, top_n=80):
    pred_fft = np.fft.fft2(data)
    pred_fft = pred_fft[:, :50]
    resample_pred_fft = signal.resample(pred_fft, new_rate, axis=1)
    amp = np.abs(resample_pred_fft)[:, :top_n]
    pha = np.angle(resample_pred_fft)[:, :top_n]
    return amp, pha


def select_pred_spectrum(data, top_n=80, select=True):
    # for one trait there 100 prediction from 100 frames
    # data: (1, 100)
    pred_fft = np.fft.fft2(data)
    if select:
        length = int(len(pred_fft[0]) / 2)
        if top_n > length:
            top_n = length
        amp = np.abs(pred_fft)[:, :top_n]
        pha = np.angle(pred_fft)[:, :top_n]
    else:
        amp = np.abs(pred_fft)
        pha = np.angle(pred_fft)
    return amp.astype("float32"), pha.astype("float32")


def gen_spectrum_data(data_path, save_to, method):
    data = torch.load(data_path)
    spec_data_ls = []
    for pred, label in tqdm(zip(data["video_frames_feat"], data["video_label"])):
        pred, label = pred.cpu(), label.cpu()
        amp_spectrum, pha_spectrum = [], []
        for one_channel in pred.T:
            amp, pha = method(one_channel.numpy()[None, :])
            amp_spectrum.append(amp)
            pha_spectrum.append(pha)
        spectrum_data = {
            "amp_spectrum": np.concatenate(amp_spectrum, axis=0),
            "pha_spectrum": np.concatenate(pha_spectrum, axis=0),
            "video_label": label.numpy()
        }
        spec_data_ls.append(spectrum_data)

    if len(spec_data_ls) > 2000:
        # separate data in case of out-of-memory issue
        torch.save(spec_data_ls[:1500], save_to.replace(".pkl", "_1.pkl"))
        print("saved [1/4]...")
        torch.save(spec_data_ls[1500: 3000], save_to.replace(".pkl", "_2.pkl"))
        print("saved [2/4]...")
        torch.save(spec_data_ls[3000: 4500], save_to.replace(".pkl", "_3.pkl"))
        print("saved [3/4]...")
        torch.save(spec_data_ls[4500:], save_to.replace(".pkl", "_4.pkl"))
        print("saved [4/4] \n  DONE.")
    else:
        torch.save(spec_data_ls, save_to)


def gen_dataset(dir, func, method, pre_fix="pred_"):
    files = [file for file in glob.glob(f"{dir}/*.pkl") if pre_fix in os.path.basename(file)]
    for file in files:
        # if "valid" in file or "test" in file:
        #     continue
        if "spectrum" in str(func):
            name = os.path.split(file)[-1].replace(pre_fix, "spectrum_").replace("output", "data")
        elif "statistic" in str(func):
            name = os.path.split(file)[-1].replace(pre_fix, "statistic_").replace("output", "data")
        else:
            raise ValueError(
                "func used in this interface should be 'statistic' or 'spectrum' method"
            )

        save_to = os.path.join(os.path.dirname(file), name)
        func(data_path=file, save_to=save_to, method=method)


if __name__ == "__main__":
    os.chdir("..")
    # dirs = glob.glob("datasets/stage_two/*_feature_output")
    # for dir in dirs:
    #     gen_dataset(
    #         dir,  # "datasets/stage_two/persemon_pred_output",
    #         func=gen_spectrum_data,
    #         method=select_pred_spectrum,
    #         pre_fix="feature_"
    #     )
    gen_dataset(
        "datasets/stage_two/deep_bimodal_reg_extract",
        func=gen_spectrum_data,
        method=select_pred_spectrum,
        pre_fix="feature_"
    )