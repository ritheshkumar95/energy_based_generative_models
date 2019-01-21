import torch
from torchvision import transforms, datasets


def inf_train_gen(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data = datasets.ImageFolder('/tmp/kumarrit/final_images', transform=transform)
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=4
    )

    while True:
        for img, labels in loader:
            yield img
