import os
import torch


def save_model(epoch, best_acc, model, optimizer, output_dir, cfg):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc
    }
    pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == (cfg.max_epoch - 1) else "checkpoint_last.pkl"
    path_checkpoint = os.path.join(output_dir, pkl_name)
    torch.save(checkpoint, path_checkpoint)
