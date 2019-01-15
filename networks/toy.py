import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, output_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, output_dim)
        )

    def forward(self, z):
        return self.main(z)


class EnergyModel(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, 1)
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x):
        out = self.main(x)
        e_x = self.norm(out)
        return e_x.squeeze(-1), out.squeeze(-1)


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, z_dim)
        )

    # def forward(self, x, z):
    #     x = torch.cat([x, z], -1)
    #     return self.main(x).squeeze(-1)

    def forward(self, x):
        return self.main(x)
