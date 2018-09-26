import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 4 * 4 * dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.expand(z).view(z.size(0), -1, 4, 4)
        return self.main(out)


class EnergyModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim // 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.expand = nn.Linear(4 * 4 * dim, 1)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out).squeeze(-1)


class StatisticsNetwork(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim // 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.expand = nn.Linear(4 * 4 * dim, z_dim)
        self.classify = nn.Sequential(
            nn.Linear(2 * z_dim, dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(dim, 1),
        )

    def forward(self, x, z):
        out = self.main(x).view(x.size(0), -1)
        out = self.expand(out)
        out = torch.cat([out, z], -1)
        return self.classify(out).squeeze(-1)
