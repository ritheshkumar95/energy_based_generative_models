import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 121),
        )

    def forward(self, z):
        return self.main(z)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(121, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.main(x).squeeze(-1)


class StatisticsNetwork(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(121 + z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x, z):
        x = torch.cat([x, z], -1)
        return self.main(x).squeeze(-1)
