import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from dpcv.config.interpret_aud_cfg import cfg
from dpcv.engine.bi_modal_trainer import AudioTrainer
from dpcv.modeling.networks.audio_interpretability_net import get_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.interpretability_audio_data import make_data_loader, norm
from dpcv.tools.exp import run


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(cfg.OUTPUT_DIR)

    data_loader = {
        "train": make_data_loader(cfg, mode="train"),
        "valid": make_data_loader(cfg, mode="valid"),
        "test": make_data_loader(cfg, mode="test")
    }

    model = get_model(cfg)
    loss_f = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = AudioTrainer(cfg, collector, logger)

    run(cfg, data_loader, model, loss_f, optimizer, scheduler, trainer, collector, logger, log_dir)


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
            "../results/interpret_aud/11-06_00-35/checkpoint_21.pkl",
            wav,
        )
