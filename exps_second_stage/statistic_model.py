import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from mlflow import log_metric, log_param, log_params


class MLP(nn.Module):
    def __init__(self, hidden_units=256):
        super(MLP, self).__init__()
        log_param("hidden_units", hidden_units)
        self.input_layer = nn.Linear(12 * 5, hidden_units)
        self.relu_1 = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(hidden_units, hidden_units)
        self.relu_2 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_units, int(hidden_units / 4))
        self.relu_3 = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(int(hidden_units / 4), 5)
        log_param("model", "MLP")

    def forward(self, x):
        x = x.reshape(-1, 60)
        x = self.input_layer(x)
        x = self.relu_1(x)
        x = self.hidden_layer_1(x)
        x = self.relu_2(x)
        x = self.hidden_layer_2(x)
        x = self.relu_3(x)
        # x = self.dropout(x)
        x = self.output_layer(x)
        return x


class StatisticConv1D(nn.Module):

    def __init__(self):
        super(StatisticConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(1, 12))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        log_param("model", "Conv1d:conv4")

    def forward(self, x):
        x = x[..., None]
        x = x.permute(0, 3, 2, 1)   # (bs, 1, 5, 12)
        x = self.conv1(x)           # (bs, 64, 5, 1)
        x = self.relu1(x)
        x = self.conv2(x)           # (bs, 128, 5, 1)
        x = self.relu2(x)
        x = self.conv3(x)           # (bs, 1, 5, 1)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.squeeze(1).squeeze()  # (bs, 5)
        return x


class SpectrumConv1D(nn.Module):

    def __init__(self):
        super(SpectrumConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(1, 7))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1, 5))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=(1, 3))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64,  out_channels=1, kernel_size=(1, 1))
        self.adp = nn.AdaptiveAvgPool1d(1)
        self.fcn = nn.Linear(80, 5)

    def forward(self, x):
        x = x[..., None]
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.squeeze(1)
        x = self.adp(x)
        x = x.squeeze()
        x = self.fcn(x)
        return x


class SpectrumData(Dataset):

    def __init__(self, data_path):
        self.sample_ls = torch.load(data_path)

    def __getitem__(self, idx):
        sample = self.sample_ls[idx]
        return {
            "amp_spectrum": sample["amp_spectrum"],
            "pha_spectrum": sample["pha_spectrum"],
            "label": sample["video_label"],
        }

    def __len__(self):
        return len(self.sample_ls)


class StatisticData(Dataset):

    def __init__(self, data_path):
        self.sample_dict = torch.load(data_path)

    def __getitem__(self, idx):
        sample = {
            "statistic": self.sample_dict["video_statistic"][idx],
            "label": self.sample_dict["video_label"][idx],
        }
        return sample

    def __len__(self):
        return len(self.sample_dict["video_label"])


class MLPTrainer:

    def __init__(self, max_epo, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.max_epo = max_epo
        self.best_acc = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.loss = nn.SmoothL1Loss()  # nn.L1Loss()  # nn.MSELoss()
        self.loss = nn.MSELoss()

    def train(self, model, data_loader, optimizer, epo):
        model.train()
        for i, data in enumerate(data_loader):
            data, label = self.data_fmt(data)
            output = model(data)
            # output = model(data["statistic"].to(self.device))
            loss = self.loss(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                acc_avg = (1 - torch.abs(output.cpu() - label.cpu())).mean().clip(min=0)
                print(
                    "TRAINING: EPO[{:0>3}/{:0>3}] ITER[{:0>3}/{:0>3}] LOSS: {:.4f} ACC: {:.4f}".format(
                        epo, self.max_epo,
                        i, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

    def valid(self, model, data_loader, epo):
        model.eval()
        with torch.no_grad():
            batch_acc_ls = []
            for i, data in enumerate(data_loader):
                data, label = self.data_fmt(data)
                output = model(data)
                # output = model(data["statistic"].to(self.device))
                batch_acc_ls.append((1 - torch.abs(output.cpu() - label.cpu())).mean(dim=0))
            epo_acc = torch.stack(batch_acc_ls, dim=0).mean().cpu().numpy()
            log_metric("valid_acc", float(epo_acc * 100))
            if epo_acc > self.best_acc:
                self.best_acc = epo_acc
                self.save_model(model, epo, self.best_acc)

            print(
                "VALID: EPO[{:0>3}/{:0>3}] ACC: {:.4f}".format(
                    epo, self.max_epo, epo_acc
                )
            )

    def test(self, model, data_loader):
        model.eval()
        with torch.no_grad():
            batch_acc_ls = []
            for i, data in enumerate(data_loader):
                data, label = self.data_fmt(data)
                output = model(data)
                batch_acc_ls.append((1 - torch.abs(output.cpu() - label.cpu())).mean(dim=0))
            epo_acc = torch.stack(batch_acc_ls, dim=0).mean().numpy()
        log_metric("test_acc", float(epo_acc * 100))
        print("TEST: ACC: {:.4f}".format(epo_acc))

    def save_model(self, model, epoch, best_acc):

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc
        }
        pkl_name = f"checkpoint_{epoch}.pkl"
        path_checkpoint = os.path.join(self.output_dir, pkl_name)
        torch.save(checkpoint, path_checkpoint)

    def data_fmt(self, data):
        return data["statistic"].to(self.device), data["label"].to(self.device)


class SpectrumTrainer(MLPTrainer):

    def data_fmt(self, data):
        return data["amp_spectrum"].to(self.device).type(torch.float32), data["label"].to(self.device)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--bs", default=64, type=int, help="batch size in training")
    parser.add_argument("--max_epoch", default=1000, type=int, help="max training epochs")
    parser.add_argument("--lr_scale_rate", default=0.1, type=float, help="learning rate scale")
    parser.add_argument("--milestones", default=[500, 800], type=list, help="where to scale learning rate")
    parser.add_argument("--output_dir", default="result_spectrum", type=str, help="where to save training output")
    args = parser.parse_args()
    return args


def main():
    log_param("exp", "swin")
    args = args_parse()
    log_params({"lr": args.lr, "epochs": args.max_epoch, "milestones": args.milestones, "bs": args.bs})

    dataset = {
        "train": "datasets/stage_two/swin_frame_pred_output/spectrum_train_data.pkl",
        "valid": "datasets/stage_two/swin_frame_pred_output/spectrum_valid_data.pkl",
        "test":  "datasets/stage_two/swin_frame_pred_output/spectrum_test_data.pkl",
    }
    train_data_loader = DataLoader(
        SpectrumData(dataset["train"]), batch_size=args.bs, shuffle=True,
        num_workers=4,
        # StatisticData(dataset["train"]), batch_size = args.bs, shuffle = True
    )
    valid_data_loader = DataLoader(
        SpectrumData(dataset["valid"]), batch_size=args.bs, shuffle=False,
        num_workers=4,
        # StatisticData(dataset["valid"]), batch_size = args.bs, shuffle = False
    )
    test_data_loader = DataLoader(
        SpectrumData(dataset["test"]), batch_size=args.bs,
        num_workers=4,
        # StatisticData(dataset["test"]), batch_size = args.bs
    )
    # model = MLP().cuda()
    # model = StatisticConv1D().cuda()
    model = SpectrumConv1D().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=args.lr_scale_rate, milestones=args.milestones)

    trainer = SpectrumTrainer(max_epo=args.max_epoch, output_dir=args.output_dir)
    # trainer = MLPTrainer(max_epo=args.max_epoch, output_dir=args.output_dir)

    for epo in range(args.max_epoch):
        trainer.train(model, train_data_loader, optimizer, epo)
        trainer.valid(model, valid_data_loader, epo)
        scheduler.step()
    trainer.test(model, test_data_loader)


if __name__ == "__main__":
    os.chdir("..")
    main()
