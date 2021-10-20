import torch
from video_dataset import VideoDataset, VideoLoader
from dpcv.data.transforms.spatial_transforms import (
    Compose,
    Normalize,
    Resize,
    CenterCrop,
    MultiScaleCornerCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    ScaleValue,
    ColorJitter,
    PickFirstChannels,
)
from dpcv.data.transforms.temporal_transforms import Compose as TemporalCompose
from dpcv.data.transforms.temporal_transforms import (
    TemporalRandomCrop,
    TemporalCenterCrop,
    TemporalSubsampling,
)


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(
    video_path,
    annotation_path,
    spatial_transform=None,
    temporal_transform=None,
):

    loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)

    training_data = VideoDataset(
        video_path,
        annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter
    )

    return training_data


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_loader(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
    train_data = VideoDataset(
        opt.video_path,
        opt.annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter
    )
    # train_data = get_training_data(
    #     opt.video_path,
    #     opt.annotation_path,
    #     spatial_transform,
    #     temporal_transform,
    # )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
    )
    return train_loader