import torch
from dpcv.engine.bi_modal_trainer import BiModalTrainer


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

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs = data["glo_img"], data["loc_img"], data["wav_aud"]
        cls_label, reg_label = data["cls_label"], data["reg_label"]
        return inputs, cls_label, reg_label

    def loss_compute(self, loss_f, reg_pred, reg_label, cls_score, cls_label, epoch_idx):
        loss_1 = loss_f["l1_loss"](reg_pred, reg_label)
        loss_2 = loss_f["mse_loss"](reg_pred, reg_label)
        loss_3 = loss_f["bell_loss"](reg_pred, reg_label)
        lambda_ = (4 * epoch_idx) / (self.cfg.MAX_EPOCH + 1)
        loss_4 = lambda_ * loss_f["ce_loss"](cls_score, cls_label)
        loss = loss_1 + loss_2 + loss_3 + loss_4
        return loss
