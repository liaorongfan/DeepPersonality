import torch
from dpcv.checkpoint.save import load_model
from dpcv.config.default_config_opt import cfg, cfg_from_file
from dpcv.experiment.exp_runner import ExpRunner
from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.datasets.video_frame_data import AllSampleFrameData2
from dpcv.data.datasets.audio_visual_data import ALLSampleAudioVisualData2
from dpcv.data.datasets.cr_data import AllFrameCRNetData
from dpcv.data.datasets.pers_emo_data import AllFramePersEmoNData2


def feature_extract(cfg_file, model_weight, data_loader, output_dir):

    cfg_from_file(cfg_file)
    cfg.MODEL.RETURN_FEATURE = True

    runner = ExpRunner(cfg)
    runner.model = load_model(runner.model, model_weight)
    # ocean_acc_avg, ocean_acc, dataset_output, dataset_label = runner.trainer.full_test(
    #     data_loader(cfg, mode="test"), runner.model
    # )
    # print(ocean_acc_avg, ocean_acc)

    for mode in ["train", "valid", "test"]:
        # note if cuda out of memory, run each mode separately
        dataloader = data_loader(cfg, mode=mode)
        output_sub_dir = os.path.join(output_dir, mode)
        runner.data_extract(dataloader, output_sub_dir)


def setup_dataloader(cfg, mode):
    assert mode in ["train", "valid", "test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    transform = build_transform_spatial(cfg)
    if mode == "test":
        data_set = AllSampleFrameData2(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
    elif mode == "train":
        data_set = AllSampleFrameData2(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = AllSampleFrameData2(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transform,
        )

    return data_set


def setup_bimodal_resnet_dataloader(cfg, mode):
    assert mode in ["train", "valid", "test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    transform = build_transform_spatial(cfg)
    if mode == "test":
        data_set = ALLSampleAudioVisualData2(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = ALLSampleAudioVisualData2(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA,
            cfg.DATA.VALID_AUD_DATA,
            cfg.DATA.VALID_LABEL_DATA,
            transform,
        )
    elif mode == "train":
        data_set = ALLSampleAudioVisualData2(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA,
            cfg.DATA.TRAIN_AUD_DATA,
            cfg.DATA.TRAIN_LABEL_DATA,
            transform,
        )

    return data_set


def setup_crnet_dataloader(cfg, mode):
    assert mode in ["train", "valid", "test"], \
        f"{mode} should be one of 'train', 'valid' or 'test'"

    transform = build_transform_spatial(cfg)
    data_cfg = cfg.DATA
    if mode == "test":
        data_set = AllFrameCRNetData(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_IMG_FACE_DATA,
            data_cfg.TEST_AUD_DATA,
            data_cfg.TEST_LABEL_DATA,
            transform
        )
    elif mode == "train":
        data_set = AllFrameCRNetData(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_IMG_FACE_DATA,
            data_cfg.TRAIN_AUD_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            transform,
        )
    elif mode == "valid":
        data_set = AllFrameCRNetData(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_IMG_FACE_DATA,
            data_cfg.VALID_AUD_DATA,
            data_cfg.VALID_LABEL_DATA,
            transform,
        )

    return data_set


def setup_persemon_dataloader(cfg, mode):
    per_trans = build_transform_spatial(cfg)
    emo_trans = build_transform_spatial(cfg)
    data_cfg = cfg.DATA
    if mode == "train":
        dataset = AllFramePersEmoNData2(
            data_cfg.ROOT,
            data_cfg.TRAIN_IMG_DATA,
            data_cfg.TRAIN_LABEL_DATA,
            data_cfg.VA_DATA,
            data_cfg.VA_TRAIN_LABEL,
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    elif mode == "valid":
        dataset = AllFramePersEmoNData2(
            data_cfg.ROOT,
            data_cfg.VALID_IMG_DATA,
            data_cfg.VALID_LABEL_DATA,
            data_cfg.VA_DATA,
            data_cfg.VA_VALID_LABEL,
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    elif mode == "test":
        dataset = AllFramePersEmoNData2(
            data_cfg.ROOT,
            data_cfg.TEST_IMG_DATA,
            data_cfg.TEST_LABEL_DATA,
            data_cfg.VA_DATA,
            data_cfg.VA_VALID_LABEL,
            per_trans=per_trans,
            emo_trans=emo_trans,
        )
    return dataset


if __name__ == "__main__":
    import os
    os.chdir("..")

    # interpret_cnn feature extract
    feature_extract(
        cfg_file="config/second_stage/statistic_on_pred/09_hrnet_statistic.yaml",
        model_weight="results/unified_frame_images/10_swin_transformer/12-13_21-28/checkpoint_110.pkl",
        data_loader=setup_dataloader,
        output_dir="datasets/second_stage/swin_transformer_extract",
    )

    # # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/08_senet.yaml",
    #     model_weight="results/unified_frame_images/08_senet/checkpoint_senet_acc_9051.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/senet_extract",
    # )

    # interpret_cnn feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/06_interpret_cnn.yaml",
    #     model_weight="results/unified_frame_images/06_interpret_cnn/checkpoint_interpret-cnn_acc_9118.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/interpret_cnn_extract",
    # )

    # # persemon feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/05_persemon.yaml",
    #     model_weight="results/unified_frame_images/05_peremon/12-23_00-07/checkpoint_160.pkl",
    #     data_loader=setup_persemon_dataloader,
    #     output_dir="datasets/second_stage/persemon_extract",
    # )

    # # crnet feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/04_crnet.yaml",
    #     model_weight="results/unified_frame_images/04_crnet/12-07_09-01/checkpoint_85.pkl",
    #     data_loader=setup_crnet_dataloader,
    #     output_dir="datasets/stage_two/cr_net_extract",
    # )

    # bimodal_renet18 feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/03_bimodal_resnet18.yaml",
    #     model_weight="results/unified_frame_images/03_bimodal_resnet/12-19_23-35/checkpoint_297.pkl",
    #     data_loader=setup_bimodal_resnet_dataloader,
    #     output_dir="datasets/second_stage/bimodal_resnet18_extract",
    # )

    # # deep_bimodal_regression feature extract
    # feature_extract(
    #     cfg_file="config/unified_frame_images/01_deep_bimodal_regression.yaml",
    #     model_weight="results/unified_frame_images/01_deep_bimodal/12-06_00-50/checkpoint_84.pkl",
    #     data_loader=setup_dataloader,
    #     output_dir="datasets/second_stage/deep_bimodal_reg_extract",
    # )
