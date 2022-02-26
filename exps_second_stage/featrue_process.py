import os
import torch
from tqdm import tqdm
import glob
import numpy as np
from scipy import signal


def gen_statistic_data(data_path, save_to):
    data = torch.load(data_path)
    statistic_data_ls = []
    for sample in data["video_frames_pred"]:
        statistic_data_ls.append(assemble_pred_statistic(sample))
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


def assemble_pred_spectrum(data, new_rate=100, top_n=80):
    # tem = np.random.randn(1, 5)
    pred_fft = np.fft.fft2(data)
    pred_fft = pred_fft[:, :50]
    resample_pred_fft = signal.resample(pred_fft, new_rate, axis=1)
    amp = np.abs(resample_pred_fft)[:, :top_n]
    pha = np.angle(resample_pred_fft)[:, :top_n]
    return amp, pha


def gen_spectrum_data(data_path, save_to):
    data = torch.load(data_path)
    spec_data_ls = []
    for pred, label in tqdm(zip(data["video_frames_pred"], data["video_label"])):
        amp_spectrum, pha_spectrum = [], []
        for one_channel in pred.T:
            amp, pha = assemble_pred_spectrum(one_channel.numpy()[None, :])
            amp_spectrum.append(amp)
            pha_spectrum.append(pha)
        spectrum_data = {
            "amp_spectrum": np.concatenate(amp_spectrum, axis=0),
            "pha_spectrum": np.concatenate(pha_spectrum, axis=0),
            "video_label": label.numpy()
        }
        spec_data_ls.append(spectrum_data)
    torch.save(spec_data_ls, save_to)


def gen_dataset(dir, func):
    files = [file for file in glob.glob(f"{dir}/*.pkl") if "pred_" in os.path.basename(file)]
    for file in files:
        if "spectrum" in str(func):
            name = os.path.split(file)[-1].replace("pred_", "spectrum_").replace("output", "data")
        elif "statistic" in str(func):
            name = os.path.split(file)[-1].replace("pred_", "statistic_").replace("output", "data")
        save_to = os.path.join(os.path.dirname(file), name)
        func(data_path=file, save_to=save_to)


if __name__ == "__main__":
    os.chdir("..")

    gen_dataset("datasets/stage_two/hrnet_pred_output", func=gen_statistic_data)
