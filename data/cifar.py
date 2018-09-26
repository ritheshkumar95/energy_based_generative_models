import torch
from torchvision import transforms, datasets


def inf_train_gen(batch_size):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data/CIFAR10', train=True, download=True,
            transform=transf
        ), batch_size=64, drop_last=True
    )
    while True:
        for img, labels in loader:
            yield img
