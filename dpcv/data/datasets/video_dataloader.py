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


def get_normalize_method(mean, std, no_mean_norm=False, no_std_norm=False):
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


def get_train_loader(video_path, annotation_path, batch_size, num_workers):
    normalize = get_normalize_method(
        [0.4345, 0.4051, 0.3775],
        [0.2768, 0.2713, 0.2737],
    )
    spatial_transform = [
        Resize(112),
        CenterCrop(112),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
    spatial_transform = Compose(spatial_transform)

    temporal_transform = [TemporalCenterCrop(16)]
    temporal_transform = TemporalCompose(temporal_transform)

    loader = VideoLoader(image_name_formatter)

    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
    train_data = VideoDataset(
        video_path,
        annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader


if __name__ == "__main__":
    opt = ""
    data_loader = get_train_loader()
    for i, (inputs, targets) in enumerate(data_loader):
        if i > 2:
            break
        print(inputs.shape)