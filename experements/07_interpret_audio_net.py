import torch
import torch.nn as nn
import os
import torch.optim as optim
from datetime import datetime
import torchaudio
from dpcv.config.interpret_dan_cfg import cfg
from dpcv.engine.bi_modal_trainer import InterpretAudioTrainer
from dpcv.modeling.networks.audio_interpretability_net import get_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.interpretability_audio_data import make_data_loader, norm


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(out_dir=os.path.join("..", "results"))
    logger.info("file_name: \n{}\n".format(__file__))

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")

    model = get_model(cfg)
    loss_f = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = InterpretAudioTrainer(cfg, collector, logger)

    if cfg.RESUME:
        model, optimizer, epoch = resume_training(cfg.RESUME, model, optimizer)
        cfg.START_EPOCH = epoch
        logger.info(f"resume training from {cfg.RESUME}")

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
        trainer.train(train_loader, model, loss_f, optimizer, epoch)
        trainer.valid(valid_loader, model, loss_f, epoch)
        scheduler.step()
        if collector.model_save:
            save_model(epoch, collector.best_valid_acc, model, optimizer, log_dir, cfg)
            collector.update_best_epoch(epoch)

    collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, log_dir)
    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
            collector.best_valid_acc,
            collector.best_epoch,
        ))


def audio_process(aud_file):
    aud_data, sample_rate = torchaudio.load(aud_file)
    trans_aud = torchaudio.transforms.Resample(sample_rate, 4000)(aud_data[0, :].view(1, -1))
    trans_fft = torch.fft.fft(trans_aud)
    half_length = int(trans_aud.shape[-1] / 2)
    trans_fre = torch.abs(trans_fft)[..., :half_length]
    trans_fre_norm = norm(trans_fre)
    if trans_fre_norm.shape[-1] < 30604:
        print("unusual input audio with the length:{}".format(trans_fre_norm.shape[-1]))
    return trans_fre_norm


def load_model(cfg, weights):
    model = get_model(cfg)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def visualize_cam(model_weights, image, trait_id=None):
    from dpcv.tools.cam import CAM
    from dpcv.tools.cam_vis import to_pil_image, overlay_audio_mask
    import matplotlib.pylab as plt
    aud_tensor = audio_process(image)
    model = load_model(cfg, model_weights)
    cam_extractor = CAM(model, "gap", enable_hooks=False, conv1d=True)
    cam_extractor._hooks_enabled = True

    model.zero_grad()
    scores = model(aud_tensor.unsqueeze(0).cuda())

    trait_id = scores.squeeze(0).argmax().item() if trait_id is None else trait_id
    activation_map = cam_extractor(trait_id, scores).cpu()

    cam_extractor.clear_hooks()
    cam_extractor._hooks_enabled = False

    # heatmap = to_pil_image(activation_map, mode='F')
    result = overlay_audio_mask(aud_tensor, activation_map)

    plt.plot(result[0])
    plt.show()


if __name__ == "__main__":
    # args = parse_args()
    # main(args, cfg)
    wav_ls = ["../datasets/raw_voice/validationData/0mym1CooiTE.005.wav",
              "../datasets/raw_voice/validationData/0uCqd5hZcyI.004.wav",
              "../datasets/raw_voice/validationData/1pm5uoU85FI.004.wav",
              "../datasets/raw_voice/validationData/2rV3Ibtdnvs.001.wav",
              "../datasets/raw_voice/validationData/5KHOpRCxnwQ.001.wav"]
    for wav in wav_ls:
        visualize_cam(
            "../results/10-05_21-03/checkpoint_21.pkl",
            wav,
        )
