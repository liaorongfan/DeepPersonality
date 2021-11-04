import torch
import torch.nn as nn
import os
import torch.optim as optim
from datetime import datetime
from dpcv.config.interpret_dan_cfg import cfg
from dpcv.engine.bi_modal_trainer import ImageModalTrainer
from dpcv.modeling.networks.interpret_dan import get_interpret_dan_model
from dpcv.tools.common import setup_seed, setup_config
from dpcv.tools.logger import make_logger
from dpcv.checkpoint.save import save_model, resume_training
from dpcv.tools.common import parse_args
from dpcv.evaluation.summary import TrainSummary
from dpcv.data.datasets.video_frame_data import make_data_loader


def main(args, cfg):
    setup_seed(12345)
    cfg = setup_config(args, cfg)
    logger, log_dir = make_logger(out_dir=os.path.join("..", "results"))
    logger.info("file_name: \n{}\n".format(__file__))

    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="valid")

    model = get_interpret_dan_model(cfg, pretrained=True)
    loss_f = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    collector = TrainSummary()
    trainer = ImageModalTrainer(cfg, collector, logger)

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


def image_process(img):
    import torchvision.transforms as transforms
    from PIL import Image
    trans_resize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
    ])
    trans_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img).convert("RGB")
    img_resize = trans_resize(img)
    img_tensor = trans_tensor(img_resize).unsqueeze(0).cuda()
    return img_resize, img_tensor


def load_model(cfg, weights):
    model = get_interpret_dan_model(cfg, pretrained=False)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(weights)
    return model


def visualize_cam(model_weights, image, trait_id=None):
    from dpcv.tools.cam import CAM
    from dpcv.tools.cam_vis import to_pil_image, overlay_mask
    import matplotlib.pylab as plt
    img_resize, img_tensor = image_process(image)
    model = load_model(cfg, model_weights)
    cam_extractor = CAM(model, enable_hooks=False)
    cam_extractor._hooks_enabled = True

    model.zero_grad()
    scores = model(img_tensor)

    trait_id = scores.squeeze(0).argmax().item() if trait_id is None else trait_id
    activation_map = cam_extractor(trait_id, scores).cpu()

    cam_extractor.clear_hooks()
    cam_extractor._hooks_enabled = False

    heatmap = to_pil_image(activation_map, mode='F')
    result = overlay_mask(img_resize, heatmap, alpha=0.5)

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    # args = parse_args()
    # main(args, cfg)
    visualize_cam(
        "../results/09-20_00-05/checkpoint_89.pkl",
        "../datasets/image_data/test_data/--Ymqszjv54.000/frame_200.jpg",
    )
