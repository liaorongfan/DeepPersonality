import torch
import torch.nn as nn
import numpy as np
from dpcv.modeling.loss.cr_loss import one_hot_CELoss, BellLoss

bell_loss = BellLoss()
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, optimizer, epochs, scheduler, device, cfg, logger):
        model.train()

        if model.train_guider:
            logger.info("Training: classification phrase")
        else:
            logger.info("Training: regression phrase")

        for epo in range(epochs):
            lambda_ = (4 * epochs) / (epo + 1)
            for i, data in enumerate(data_loader):
                for k, v in data.items():
                    data[k] = v.to(device)

                # forward
                if model.train_guider:
                    cls_score = model(data["glo_img"], data["loc_img"], data["wav_aud"])
                    loss = one_hot_CELoss(cls_score, data["cls_label"])
                else:
                    cls_score, reg_pred = model(data["glo_img"], data["loc_img"], data["wav_aud"])
                    loss_1 = l1_loss(reg_pred, data["reg_label"])
                    loss_2 = mse_loss(reg_pred, data["reg_label"])
                    loss_3 = bell_loss(reg_pred, data["reg_label"])
                    loss_4 = lambda_ * one_hot_CELoss(cls_score, data["cls_label"])
                    # loss_4 = one_hot_CELoss(cls_score, data["cls_label"])
                    loss = loss_1 + loss_2 + loss_3 + loss_4
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss)

                if i % cfg.LOG_INTERVAL == cfg.LOG_INTERVAL - 1:
                    if model.train_guider:
                        cls_soft_max = torch.softmax(cls_score, dim=-1)
                        matched = torch.as_tensor(
                            torch.argmax(cls_soft_max, -1) == torch.argmax(data["cls_label"], -1),
                            dtype=torch.int8
                        )
                        acc = matched.sum() / matched.numel()
                    else:
                        acc = (1 - torch.abs(reg_pred - data["reg_label"])).mean(dim=0).clip(min=0)

                    logger.info(
                        "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".
                        format(epo + 1, cfg.MAX_EPOCH, i + 1, len(data_loader),
                               float(loss.cpu().detach().numpy()),
                               acc.cpu().detach().numpy().round(2))
                    )
                # acc_avg = acc_avg.detach().numpy()
                # if acc_avg < 0:
                #     acc_avg = 0
                # acc_avg_list.append(acc_avg)
                # print loss info for an interval
                # debug
                # if i > 10:
                #     break
            logger.info("epoch:{}".format(1))

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()
        with torch.no_grad():
            loss_list = []
            loss_mean = 0
            acc_batch_list = []
            for i, data in enumerate(data_loader):
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_f(outputs.cpu(), labels.cpu())
                loss_list.append(loss.item())
                loss_mean = np.mean(loss_list)
                acc_batch_list.append((1 - np.abs(outputs.cpu().detach().numpy() - labels.cpu().detach().numpy())))
                if i % 50 == 0:
                    print(f"computing {i} batches ...")
            ocean_acc = np.concatenate(acc_batch_list, axis=0).mean(axis=0)
            acc_avg = ocean_acc.mean()

        return loss_mean, ocean_acc, acc_avg
