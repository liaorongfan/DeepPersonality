import torch
import os
import torch.nn as nn
from mlflow import log_metric, log_param, log_params


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
        return epo_acc

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

