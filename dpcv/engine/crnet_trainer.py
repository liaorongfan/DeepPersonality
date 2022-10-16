import torch
from tqdm import tqdm
from dpcv.engine.bi_modal_trainer import BiModalTrainer
import numpy as np
import math
import os
from .build import TRAINER_REGISTRY


class CRNetTrainer(BiModalTrainer):

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()

        if model.train_guider:
            self.logger.info(f"Training: classification phrase, learning rate:{optimizer[0].param_groups[0]['lr']}")
        else:
            self.logger.info(f"Training: regression phrase, learning rate:{optimizer[1].param_groups[0]['lr']}")

        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, cls_label, reg_label = self.data_fmt(data)
            # forward
            if model.train_guider:
                cls_score = model(*inputs)
                loss = loss_f["ce_loss"](cls_score, cls_label)
                optimizer[0].zero_grad()
                loss.backward()
                optimizer[0].step()
            else:
                cls_score, reg_pred = model(*inputs)
                loss = self.loss_compute(loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx)
                # backward
                optimizer[1].zero_grad()
                loss.backward()
                optimizer[1].step()

                loss_list.append(loss.item())
                acc_avg = (1 - torch.abs(reg_pred.cpu() - reg_label.cpu())).mean().clip(min=0)
                acc_avg = acc_avg.detach().numpy()
                acc_avg_list.append(acc_avg)

            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                if model.train_guider:
                    cls_soft_max = torch.softmax(cls_score, dim=-1)
                    matched = torch.as_tensor(
                        torch.argmax(cls_soft_max, -1) == torch.argmax(cls_label, -1),
                        dtype=torch.int8
                    )
                    acc = matched.sum() / matched.numel()
                    self.logger.info(
                        "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".
                        format(epoch_idx, self.cfg.TRAIN_CLS_EPOCH, i + 1, len(data_loader),
                               float(loss.cpu().detach().numpy()),
                               acc.cpu().detach().numpy().round(2))
                    )

                else:
                    acc = (1 - torch.abs(reg_pred - reg_label)).mean(dim=0).clip(min=0)

                    self.logger.info(
                        "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".
                        format(epoch_idx, self.cfg.MAX_EPOCH, i + 1, len(data_loader),
                               float(loss.cpu().detach().numpy()),
                               acc.cpu().detach().numpy().round(2))
                    )

        if not model.train_guider:
            self.clt.record_train_loss(loss_list)
            self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        if not model.train_guider:
            with torch.no_grad():
                loss_batch_list = []
                acc_batch_list = []
                ocean_acc_epoch = []
                for i, data in enumerate(data_loader):
                    inputs, cls_label, reg_label = self.data_fmt(data)
                    cls_score, reg_pred = model(*inputs)
                    loss = self.loss_compute(loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx)
                    loss_batch_list.append(loss.item())
                    ocean_acc_batch = (1 - torch.abs(reg_pred.cpu() - reg_label.cpu())).mean(dim=0).clip(min=0)
                    ocean_acc_epoch.append(ocean_acc_batch)
                    acc_batch_avg = ocean_acc_batch.mean()
                    acc_batch_list.append(acc_batch_avg)
                ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
                ocean_acc_avg = ocean_acc.mean()

            self.clt.record_valid_loss(loss_batch_list)  # acc over batches
            self.clt.record_valid_acc(acc_batch_list)  # acc over batches
            self.clt.record_valid_ocean_acc(ocean_acc)

            if ocean_acc_avg > self.clt.best_valid_acc:
                self.clt.update_best_acc(ocean_acc_avg)
                self.clt.update_model_save_flag(1)
            else:
                self.clt.update_model_save_flag(0)

        else:
            print("only test regression accuracy")

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
                inputs, _, labels = self.data_fmt(data)
                _, outputs = model(*inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                ocean_acc.append(ocean_acc_batch)
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.stack(output_list, dim=0).view(-1, 5).numpy()
            dataset_label = torch.stack(label_list, dim=0).view(-1, 5).numpy()

        return ocean_acc_avg, ocean_acc, dataset_output, dataset_label

    def full_test(self, data_loader, model):
        model.eval()
        with torch.no_grad():

            for data in tqdm(data_loader):
                inputs, labels = self.full_test_data_fmt(data)
                _, outputs = model(*inputs)

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs = data["glo_img"], data["loc_img"], data["wav_aud"]
        cls_label, reg_label = data["cls_label"], data["reg_label"]
        return inputs, cls_label, reg_label

    def full_test_data_fmt(self, data):
        inputs = data["glo_img"].to(self.device), data["loc_img"].to(self.device), data["wav_aud"].to(self.device)
        label = data["reg_label"]
        return inputs, label

    def loss_compute(self, loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx):
        loss_1 = loss_f["l1_loss"](reg_pred, reg_label)
        loss_2 = loss_f["mse_loss"](reg_pred, reg_label)
        loss_3 = loss_f["bell_loss"](reg_pred, reg_label)
        lambda_ = (4 * epoch_idx) / (self.cfg.MAX_EPOCH + 1)
        loss_4 = lambda_ * loss_f["ce_loss"](cls_score, cls_label)
        loss = loss_1 + loss_2 + loss_3 + loss_4
        return loss


@TRAINER_REGISTRY.register()
class CRNetTrainer2(BiModalTrainer):

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()
        if epoch_idx > model.train_guider_epo:
            model.set_train_regressor()

        if not model.train_regressor:
            self.logger.info(f"Training: classification phrase, learning rate:{optimizer[0].param_groups[0]['lr']}")
        else:
            self.logger.info(f"Training: regression phrase, learning rate:{optimizer[1].param_groups[0]['lr']}")

        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, cls_label, reg_label = self.data_fmt(data)
            # forward
            if not model.train_regressor:
                cls_score = model(*inputs)
                loss = loss_f["ce_loss"](cls_score, cls_label)
                optimizer[0].zero_grad()
                loss.backward()
                optimizer[0].step()
            else:
                cls_score, reg_pred = model(*inputs)
                loss = self.loss_compute(loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx)
                # backward
                optimizer[1].zero_grad()
                loss.backward()
                optimizer[1].step()

                loss_list.append(loss.item())
                acc_avg = (1 - torch.abs(reg_pred.cpu() - reg_label.cpu())).mean().clip(min=0)
                # acc_avg = acc_avg.detach().numpy()
                acc_avg_list.append(acc_avg.detach().numpy())

            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                if not model.train_regressor:
                    cls_soft_max = torch.softmax(cls_score, dim=-1)
                    matched = torch.as_tensor(
                        torch.argmax(cls_soft_max, -1) == torch.argmax(cls_label, -1),
                        dtype=torch.int8
                    )
                    acc = matched.sum() / matched.numel()
                    self.logger.info(
                        "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".
                        format(epoch_idx, model.train_guider_epo, i + 1, len(data_loader),
                               float(loss.cpu().detach().numpy()),
                               acc.cpu().detach().numpy().round(2))
                    )

                else:
                    acc = (1 - torch.abs(reg_pred - reg_label)).mean(dim=0).clip(min=0)

                    self.logger.info(
                        "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".
                        format(epoch_idx, self.cfg.MAX_EPOCH, i + 1, len(data_loader),
                               float(loss.cpu().detach().numpy()),
                               acc.cpu().detach().numpy().round(2))
                    )

        if model.train_regressor:
            self.clt.record_train_loss(loss_list)
            self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        if model.train_regressor:
            with torch.no_grad():
                loss_batch_list = []
                acc_batch_list = []
                ocean_acc_epoch = []
                for i, data in enumerate(data_loader):
                    inputs, cls_label, reg_label = self.data_fmt(data)
                    cls_score, reg_pred = model(*inputs)
                    loss = self.loss_compute(loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx)
                    loss_batch_list.append(loss.item())
                    ocean_acc_batch = (1 - torch.abs(reg_pred.cpu() - reg_label.cpu())).mean(dim=0).clip(min=0)
                    ocean_acc_epoch.append(ocean_acc_batch)
                    acc_batch_avg = ocean_acc_batch.mean()
                    acc_batch_list.append(acc_batch_avg)
                ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
                ocean_acc_avg = ocean_acc.mean()

            self.clt.record_valid_loss(loss_batch_list)  # acc over batches
            self.clt.record_valid_acc(acc_batch_list)  # acc over batches
            self.clt.record_valid_ocean_acc(ocean_acc)

            if ocean_acc_avg > self.clt.best_valid_acc:
                self.clt.update_best_acc(ocean_acc_avg)
                self.clt.update_model_save_flag(1)
            else:
                self.clt.update_model_save_flag(0)

        else:
            print("only test regression accuracy")
            return

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
        model.set_train_regressor()
        mse_func = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            mse_ls = []
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, cls_label, labels = self.data_fmt(data)
                _, outputs = model(*inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                mse = mse_func(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                mse_ls.append(mse)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(mse_ls, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_mean = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.stack(output_list, dim=0).view(-1, 5).numpy()
            dataset_label = torch.stack(label_list, dim=0).view(-1, 5).numpy()

        ocean_mse_mean_rand = np.round(ocean_mse_mean, 4)
        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg, ocean_acc, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_mean_rand)

    def full_test(self, data_loader, model):
        model.eval()
        model.set_train_regressor()
        with torch.no_grad():
            out_ls, label_ls = [], []
            for data in tqdm(data_loader):
                inputs, labels = self.full_test_data_fmt(data)
                _, outputs = model(*inputs)
                out_ls.append(outputs.mean(0).cpu().detach())
                label_ls.append(labels)
            all_out = torch.stack(out_ls, 0)
            all_label = torch.stack(label_ls, 0)
            ocean_acc = (1 - torch.abs(all_out - all_label)).mean(0).numpy()
            ocean_acc_avg = ocean_acc.mean(0)

            ocean_acc_avg_rand = np.round(ocean_acc_avg, 4)
            ocean_acc_dict = {k: np.round(ocean_acc[i], 4) for i, k in enumerate(["O", "C", "E", "A", "N"])}

            dataset_output = all_out.numpy()
            dataset_label = all_label.numpy()

            return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs = data["glo_img"], data["loc_img"], data["wav_aud"]
        cls_label, reg_label = data["cls_label"], data["reg_label"]
        return inputs, cls_label, reg_label

    def full_test_data_fmt(self, data):
        glo_imgs = torch.stack(data["glo_img"], 0).to(self.device)
        loc_imgs = torch.stack(data["loc_img"], 0).to(self.device)
        wav_aud = data["wav_aud"].repeat(len(glo_imgs), 1, 1, 1).to(self.device)

        inputs = glo_imgs, loc_imgs, wav_aud
        label = data["reg_label"]
        return inputs, label

    def loss_compute(self, loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx):
        loss_1 = loss_f["l1_loss"](reg_pred, reg_label)
        loss_2 = loss_f["mse_loss"](reg_pred, reg_label)
        loss_3 = loss_f["bell_loss"](reg_pred, reg_label)
        lambda_ = (4 * epoch_idx) / (self.cfg.MAX_EPOCH + 1)
        loss_4 = lambda_ * loss_f["ce_loss"](cls_score, cls_label)
        loss = loss_1 + loss_2 + loss_3 + loss_4
        return loss

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        model.set_train_regressor()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                mini_batch_size = 16
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / mini_batch_size)):
                    mini_batch_i_1 = inputs[0][(i * mini_batch_size): (i + 1) * mini_batch_size]
                    mini_batch_i_2 = inputs[1][(i * mini_batch_size): (i + 1) * mini_batch_size]
                    mini_batch_i_3 = inputs[2][(i * mini_batch_size): (i + 1) * mini_batch_size]
                    mini_batch_i = (mini_batch_i_1, mini_batch_i_2, mini_batch_i_3)
                    if model.return_feature:
                        _, out, feat = model(*mini_batch_i)
                        out_ls.append(out.cpu())
                        feat_ls.append(feat.cpu())
                    else:
                        _, out = model(*mini_batch_i)
                        out_ls.append(out.cpu())
                        feat_ls.append(torch.tensor([0]))

                # out.shape = (64, 5) feat.shape = (64, 5, 512)
                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)


@TRAINER_REGISTRY.register()
class CRNetTrainer2Vis(CRNetTrainer2):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs = data["glo_img"], data["loc_img"]
        cls_label, reg_label = data["cls_label"], data["reg_label"]
        return inputs, cls_label, reg_label


@TRAINER_REGISTRY.register()
class CRNetAudTrainer(CRNetTrainer2):
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs = data["aud_data"]
        cls_label, reg_label = data["aud_label_cls"], data["aud_label"]
        return (inputs,), cls_label, reg_label