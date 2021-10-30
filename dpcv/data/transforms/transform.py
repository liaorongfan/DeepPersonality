"""
transform operation for different networks
"""


def set_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_tpn_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_vat_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_lstm_transform():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_crnet_transform():
    import torchvision.transforms as transforms
    # norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    # norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((112, 112))
        # transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_audio_visual_transform():
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        # transforms.RandomVerticalFlip(0.5),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])
    return transforms


def set_per_transform():
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
    ])
    return transforms

