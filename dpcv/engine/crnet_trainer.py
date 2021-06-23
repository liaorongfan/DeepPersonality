import torch
import numpy as np


class ModelTrainer(object):

    @staticmethod
    def train_classification(data_loader, model, loss_func, optimizer, scheduler, device, cfg, logger):
        model.train_guide = True
        model.train()

        loss_list = []
        acc_avg_list = []
        loss_mean = 0
        acc_avg = 0
        for i, data in enumerate(data_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            glo_img, loc_img, wav_aud = data["glo_img"], data["loc_img"], data["wav_aud"]
            # glo_img, loc_img, wav_aud = glo_img.to(device), loc_img.to(device), wav_aud.to(device)

            cls_label, reg_label = data["cls_label"], data["reg_label"]

            # forward & backward
            cls_score = model(glo_img, loc_img, wav_aud)
            optimizer.zero_grad()

            loss = loss_func(cls_score, cls_label)
            loss.backward()
            optimizer.step()
            print(loss)
            # collect loss
            loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)

            # acc_avg = (1 - torch.abs(outputs.cpu() - loc_img.cpu())).mean()
            # acc_avg = acc_avg.detach().numpy()
            # if acc_avg < 0:
            #     acc_avg = 0
            # acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % cfg.LOG_INTERVAL == cfg.LOG_INTERVAL - 1:
                logger.info(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2}".
                    format(1 + 1, cfg.MAX_EPOCH, i + 1, len(data_loader), float(loss_mean), float(acc_avg))
                )
            # debug
            # if i > 10:
            #     break
        logger.info("epoch:{}".format(1))
        acc_avg = np.mean(acc_avg_list)  # return average accuracy of this training epoch


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


