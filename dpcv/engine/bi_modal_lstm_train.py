import torch
from dpcv.engine.bi_modal_trainer import BiModalTrainer


# class BiModalTrainer_(object):
#     def __init__(self, collector):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.clt = collector
#
#     def train(self, data_loader, model, loss_f, optimizer, scheduler, epoch_idx, cfg, logger):
#         model.train()
#
#         loss_list = []
#         acc_avg_list = []
#         for i, data in enumerate(data_loader):
#             img_in, aud_in, labels = self.data_fmt(data)
#             outputs = model(aud_in, img_in)
#             optimizer.zero_grad()
#             loss = loss_f(outputs.cpu(), labels.cpu())
#             loss.backward()
#             optimizer.step()
#
#             loss_list.append(loss.item())
#
#             acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
#             acc_avg = acc_avg.detach().numpy()
#             # acc_avg = 0
#             acc_avg_list.append(acc_avg)
#             # print loss info for an interval
#             if i % cfg.LOG_INTERVAL == cfg.LOG_INTERVAL - 1:
#                 logger.info(
#                     "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2}".format(
#                         epoch_idx + 1, cfg.MAX_EPOCH, i + 1, len(data_loader), float(loss.item()), float(acc_avg))
#                 )  # print current training info of that batch
#
#         self.clt.record_train_loss(loss_list)
#         self.clt.record_train_acc(acc_avg_list)
#
#     def valid(self, data_loader, model, loss_f):
#         model.eval()
#         with torch.no_grad():
#             loss_list = []
#             acc_batch_list = []
#             ocean_list = []
#             for i, data in enumerate(data_loader):
#                 img_in, aud_in, labels = self.data_fmt(data)
#                 outputs = model(aud_in, img_in)
#                 loss = loss_f(outputs.cpu(), labels.cpu())
#                 loss_list.append(loss.item())
#                 acc_batch = (1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())).mean(dim=0)
#                 ocean_list.append(acc_batch)
#                 acc_batch_avg = acc_batch.mean()
#                 acc_batch_list.append(acc_batch_avg)
#             ocean_acc = torch.stack(ocean_list, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
#             ocean_acc_avg = ocean_acc.mean()
#
#         self.clt.record_valid_loss(loss_list)
#         self.clt.record_valid_acc(acc_batch_list)  # acc over batches
#         self.clt.record_valid_ocean_acc(ocean_acc)
#         if ocean_acc_avg > self.clt.best_acc:
#             self.clt.update_best_acc(ocean_acc_avg)
#             self.clt.update_model_save_flag(1)
#         else:
#             self.clt.update_model_save_flag(0)
#
#     def data_fmt(self, data):
#         for k, v in data.items():
#             data[k] = v.to(self.device)
#         img_in, aud_in, labels = data["image"], data["audio"], data["label"]
#         img_in = img_in.view(-1, 3, 112, 112)
#         aud_in = aud_in.view(-1, 68)
#         return img_in, aud_in, labels


# class BimodalLSTMTrain(BiModalTrainer):
#
#     def data_fmt(self, data):
#         for k, v in data.items():
#             data[k] = v.to(self.device)
#         img_in, aud_in, labels = data["image"], data["audio"], data["label"]
#         img_in = img_in.view(-1, 3, 112, 112)
#         aud_in = aud_in.view(-1, 68)
#         return img_in, aud_in, labels