from dpcv.data.transforms.build import build_transform_spatial
from dpcv.data.datasets.video_frame_data import AllSampleFrameData2
from dpcv.data.datasets.audio_visual_data import ALLSampleAudioVisualData2
from dpcv.data.datasets.cr_data import AllFrameCRNetData
from dpcv.data.datasets.pers_emo_data import AllFramePersEmoNData2


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
