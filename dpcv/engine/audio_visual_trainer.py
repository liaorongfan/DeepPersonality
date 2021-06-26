import torch
import numpy as np
from collections import Counter


class BiModalTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        loss_list = []
        acc_avg_list = []
        loss_mean = 0
        acc_avg = 0
        for i, data in enumerate(data_loader):

            img_in, aud_in, labels = data["image"], data["audio"], data["label"]
            img_in, aud_in, labels = img_in.to(device), aud_in.to(device), labels.to(device)

            # forward & backward
            outputs = model(img_in, aud_in)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()
            print(loss)
            # collect loss
            loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)

            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean()
            acc_avg = acc_avg.detach().numpy()
            acc_avg = acc_avg if acc_avg > 0 else 0  # do not concern negative value
            # acc_avg = 0
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % cfg.LOG_INTERVAL == cfg.LOG_INTERVAL - 1:
                logger.info(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2}".
                    format(epoch_idx + 1, cfg.MAX_EPOCH, i + 1, len(data_loader), float(loss_mean), float(acc_avg))
                )

        logger.info("epoch:{}".format(epoch_idx))
        acc_avg = np.mean(acc_avg_list)  # return average accuracy of this training epoch
        return loss_mean, acc_avg, loss_list, acc_avg_list

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


