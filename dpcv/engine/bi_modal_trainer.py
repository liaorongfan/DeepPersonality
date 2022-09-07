import torch
from tqdm import tqdm
import numpy as np
import math
import os
from .build import TRAINER_REGISTRY
from torch.utils.tensorboard import SummaryWriter
import time


@TRAINER_REGISTRY.register()
class BiModalTrainer(object):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        self.tb_writer.add_scalar("lr", lr, epoch_idx)

        model.train()
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            epo_iter_num = len(data_loader)
            iter_start_time = time.time()

            inputs, labels = self.data_fmt(data)
            outputs = model(*inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            self.tb_writer.add_scalar("loss", loss.item(), i)
            loss.backward()
            optimizer.step()

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)

            # print loss and training info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                remain_iter = epo_iter_num - i
                remain_epo = self.cfg.MAX_EPOCH - epoch_idx
                eta = (epo_iter_num * iter_time) * remain_epo + (remain_iter * iter_time)
                eta = int(eta)
                eta_string = f"{eta // 3600}h:{eta % 3600 // 60}m:{eta % 60}s"
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] LOSS: {:.4f} ACC:{:.4f} ETA:{} ".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                        i + 1, epo_iter_num,                     # Iter  
                        iter_time,                      # IterTime
                        float(loss.item()), float(acc_avg),  # LOSS ACC ETA
                        eta_string,    
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
                ocean_acc_batch = (
                    1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())
                ).mean(dim=0).clip(min=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            self.tb_writer.add_scalar("valid_acc", ocean_acc_avg, epoch_idx)
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
        mse_func = torch.nn.MSELoss(reduction="none")
        model.eval()
        with torch.no_grad():
            mse_ls = []
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
                mse = mse_func(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0).clip(min=0)
                mse_ls.append(mse)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(mse_ls, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_mean = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()
        ocean_mse_mean_rand = np.round(ocean_mse_mean, 4)
        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        self.tb_writer.add_scalar("test_acc", ocean_acc_avg_rand)
        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)

        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_mean_rand)

    def full_test(self, data_set, model):
        model.eval()
        out_ls, label_ls = [], []
        with torch.no_grad():
            for data in tqdm(data_set):
                inputs, label = self.full_test_data_fmt(data)
                out = model(*inputs)
                out_ls.append(out.mean(0).cpu().detach())
                label_ls.append(label)
        all_out = torch.stack(out_ls, 0)
        all_label = torch.stack(label_ls, 0)
        ocean_acc = (1 - torch.abs(all_out - all_label)).mean(0).numpy()
        ocean_acc_avg = ocean_acc.mean(0)

        ocean_acc_avg_rand = np.round(ocean_acc_avg, 4)
        ocean_acc_dict = {k: np.round(ocean_acc[i], 4) for i, k in enumerate(["O", "C", "E", "A", "N"])}

        dataset_output = all_out.numpy()
        dataset_label = all_label.numpy()

        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                # mini_batch = 64
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / 64)):
                    mini_batch_1 = inputs[0][(i * 64): (i + 1) * 64]
                    mini_batch = (mini_batch_1,)
                    try:
                        mini_batch_2 = inputs[1][(i * 64): (i + 1) * 64]
                        mini_batch = (mini_batch_1, mini_batch_2)
                    except IndexError:
                        pass

                    # mini_batch = (mini_batch_1, mini_batch_2)
                    if model.return_feature:
                        out, feat = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(feat.cpu())
                    else:
                        out = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(torch.tensor([0]))
                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        return (aud_in, img_in), labels

    def full_test_data_fmt(self, data):
        images, wav, label = data["image"], data["audio"], data["label"]
        images_in = torch.stack(images, 0).to(self.device)
        # wav_in = torch.stack([wav] * 100, 0).to(self.device)
        wav_in = wav.repeat(len(images), 1, 1, 1).to(self.device)
        return (wav_in, images_in), label


@TRAINER_REGISTRY.register()
class BimodalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        aud_in = aud_in.view(-1, 68)
        return (aud_in, img_in), labels


@TRAINER_REGISTRY.register()
class BimodalLSTMTrainVisual(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in,  labels = data["image"],  data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        # aud_in = aud_in.view(-1, 68)
        return (img_in,), labels


@TRAINER_REGISTRY.register()
class ImgModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, _, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        return (img_in,), labels


@TRAINER_REGISTRY.register()
class AudModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        _, aud_in, labels = data["image"], data["audio"], data["label"]
        aud_in = aud_in.view(-1, 68)
        return (aud_in,), labels


@TRAINER_REGISTRY.register()
class DeepBimodalTrain(BimodalLSTMTrain):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels


@TRAINER_REGISTRY.register()
class ImageModalTrainer(BiModalTrainer):
    """
    for model only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels

    def full_test_data_fmt(self, data):
        images, label = data["all_images"], data["label"]
        images_in = torch.stack(images, 0).to(self.device)
        return (images_in, ), label


@TRAINER_REGISTRY.register()
class MultiModalTrainer(BiModalTrainer):
    """
    for model only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["feature"], data["label"]
        return (inputs,), labels


@TRAINER_REGISTRY.register()
class ImageListTrainer(BiModalTrainer):
    """
    for interpret cnn model, only image data used
    """
    def data_fmt(self, data):
        inputs, labels = data["image"], data["label"]
        inputs = [item.to(self.device) for item in inputs]
        labels = labels.to(self.device)
        return (inputs,), labels

    def full_test_data_fmt(self, data):
        images, label = data["all_images"], data["label"]
        # short_sque, long_sque = zip(*images)
        inputs = [torch.stack(sque, 0).to(self.device) for sque in zip(*images)]
        return (inputs,), label


@TRAINER_REGISTRY.register()
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

    def test(self, data_loader, model):
        model.eval()
        mse = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            ocean_acc = []
            ocean_mse = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                loss, outputs = model(**inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                ocean_mse_batch = mse(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                ocean_mse.append(ocean_mse_batch)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(ocean_mse, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_avg = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()

            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()
        ocean_mse_avg_rand = np.round(ocean_mse_avg.astype("float64"), 4)
        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_avg_rand)

    def full_test(self, data_loader, model):
        model.eval()
        with torch.no_grad():
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.full_test_data_fmt(data)
                loss, outputs = model(**inputs)

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

    def full_test_data_fmt(self, data):
        inputs, labels = data["all_images"], data["label"]
        inputs = torch.stack(inputs, 0).to(self.device)
        labels_repeats = labels.repeat(6, 1).to(self.device)
        data_input = {"num_modalities": [1], "img_group_0": inputs, "img_meta": None, "gt_label": labels_repeats}
        return data_input, labels_repeats


@TRAINER_REGISTRY.register()
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
        mse = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            ocean_acc = []
            ocean_mse = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, p_labels, e_labels = self.data_fmt(data)
                p_score, p_co, e_score, e_co, x_ep = model(*inputs)
                p_score = p_score.cpu().detach()
                p_labels = p_labels.cpu().detach()
                output_list.append(p_score)
                label_list.append(p_labels)
                ocean_mse_batch = mse(p_score, p_labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(p_score - p_labels)).mean(dim=0)
                ocean_mse.append(ocean_mse_batch)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(ocean_mse, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_avg = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()

            dataset_output = torch.stack(output_list, dim=0).view(-1, 5).numpy()
            dataset_label = torch.stack(label_list, dim=0).view(-1, 5).numpy()

        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_avg)

    def full_test(self, data_loader, model):
        return self.test(data_loader, model)

    def full_test_data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.squeeze().to(self.device)
        per_inputs, emo_inputs = data["per_img"], data["emo_img"],
        per_labels, emo_labels = data["per_label"], data["emo_label"]
        return (per_inputs, emo_inputs), per_labels[0]

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                mini_batch = 64
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / mini_batch)):
                    mini_batch_1 = inputs[0][(i * mini_batch): (i + 1) * mini_batch]
                    mini_batch_2 = inputs[1][i * 3: (i * 3 + mini_batch)]  # jump 3 images every time
                    mini_batch_input = (mini_batch_1, mini_batch_2)
                    out, *_, feat = model(*mini_batch_input)
                    out_ls.append(out.cpu())
                    feat_ls.append(feat.cpu())

                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)


@TRAINER_REGISTRY.register()
class AudioTrainer(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        return (data["aud_data"],), data["aud_label"]


@TRAINER_REGISTRY.register()
class StatisticTrainer(BiModalTrainer):

    def data_fmt(self, data):
        return (data["data"].to(self.device),), data["label"].to(self.device)


@TRAINER_REGISTRY.register()
class SpectrumTrainer(BiModalTrainer):

    def data_fmt(self, data):
        return (data["data"].to(self.device),), data["label"].to(self.device)
