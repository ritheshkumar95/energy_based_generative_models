import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, input_dim, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 2 * 2 * dim)
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim // 2, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 5, 2, 2),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, input_dim, 5, 2, 2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z):
        x = self.expand(z).view(z.size(0), -1, 2, 2)
        return self.main(x)


class EnergyModel(nn.Module):
    def __init__(self, input_dim, dim=512):
        super().__init__()
        self.expand = nn.Linear(2 * 2 * dim, 1)
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out).squeeze(-1)


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.expand = nn.Linear(2 * 2 * dim, z_dim)
        self.classify = nn.Sequential(
            nn.Linear(z_dim * 2, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, x, z):
        out = self.main(x).view(x.size(0), -1)
        out = self.expand(out)
        out = torch.cat([out, z], -1)
        return self.classify(out).squeeze(-1)
