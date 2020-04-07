from torchvision import transforms


def get_transforms():
    """
    Returns:
        dataset_transforms: dict, Transforms for training and testing datasets
            based on the standard deviation and means of the channels in the
            ImageNet dataset. Training dataset transforms is to be accessed
            with the use of the 'train' key, and testing dataset transform
            witht the 'test' key.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return {'train': train, 'test': test}
