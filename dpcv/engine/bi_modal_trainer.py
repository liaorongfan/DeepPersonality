import torch
from tqdm import tqdm
import numpy as np


class BiModalTrainer(object):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()
        self.logger.info(f"Training: learning rate:{optimizer.param_groups[0]['lr']}")
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, labels = self.data_fmt(data)
            outputs = model(*inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4f}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,
                        i + 1, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                loss = loss_f(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())).mean(dim=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        model.eval()
        with torch.no_grad():
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                ocean_acc.append(ocean_acc_batch)
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()

        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        keys = ["O", "C", "E", "A", "N"]
        ocean_acc_dict = {}
        for i, k in enumerate(keys):
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        return (aud_in, img_in), labels


class BimodalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        aud_in = aud_in.view(-1, 68)
        return (aud_in, img_in), labels


class ImgModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, _, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        return (img_in,), labels


class AudModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        _, aud_in, labels = data["image"], data["audio"], data["label"]
        aud_in = aud_in.view(-1, 68)
        return (aud_in,), labels


class DeepBimodalTrain(BimodalLSTMTrain):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels


class ImageModalTrainer(BiModalTrainer):
    """
    for model only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels


class ImageListTrainer(BiModalTrainer):
    """
    for interpret cnn model, only image data used
    """
    def data_fmt(self, data):
        inputs, labels = data["image"], data["label"]
        inputs = [item.to(self.device) for item in inputs]
        labels = labels.to(self.device)
        return (inputs,), labels


class TPNTrainer(BiModalTrainer):
    """
    for interpret cnn model, only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        data_input = {"num_modalities": [1], "img_group_0": inputs, "img_meta": None, "gt_label": labels}

        return data_input, labels

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()
        self.logger.info(f"Training: learning rate:{optimizer.param_groups[0]['lr']}")
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, labels = self.data_fmt(data)
            loss, outputs = model(**inputs)
            optimizer.zero_grad()
            # loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4f}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,
                        i + 1, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                loss, outputs = model(**inputs)
                # loss = loss_f(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())).mean(dim=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )


class PersEmoTrainer(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.squeeze().to(self.device)
        per_inputs, emo_inputs = data["per_img"], data["emo_img"],
        per_labels, emo_labels = data["per_label"], data["emo_label"]
        return (per_inputs, emo_inputs), per_labels, emo_labels

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()
        self.logger.info(f"Training: learning rate:{optimizer.param_groups[0]['lr']}")
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, p_labels, e_labels = self.data_fmt(data)
            p_score, p_co, e_score, e_co, x_ep = model(*inputs)
            optimizer.zero_grad()
            loss = loss_f(p_score, p_labels, e_score, e_labels, p_co, e_co, x_ep)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(p_score.cpu() - p_labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4f}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,
                        i + 1, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, p_labels, e_labels = self.data_fmt(data)
                p_score, p_co, e_score, e_co, x_ep = model(*inputs)
                loss = loss_f(p_score, p_labels, e_score, e_labels, p_co, e_co, x_ep)
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(p_score.cpu().detach() - p_labels.cpu().detach())).mean(dim=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        model.eval()
        with torch.no_grad():
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):

                inputs, p_labels, e_labels = self.data_fmt(data)
                p_score, p_co, e_score, e_co, x_ep = model(*inputs)
                p_score = p_score.cpu().detach()
                p_labels = p_labels.cpu().detach()
                output_list.append(p_score)
                label_list.append(p_labels)
                ocean_acc_batch = (1 - torch.abs(p_score - p_labels)).mean(dim=0)
                ocean_acc.append(ocean_acc_batch)
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.stack(output_list, dim=0).view(-1, 5).numpy()
            dataset_label = torch.stack(label_list, dim=0).view(-1, 5).numpy()
        return ocean_acc_avg, ocean_acc, dataset_output, dataset_label


class InterpretAudioTrainer(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        return (data["aud_data"],), data["aud_label"]
