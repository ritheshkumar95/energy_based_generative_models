import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.05)
        T.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.05)
    #     m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, output_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(output_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, z_dim)
        )
        # self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


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
        # self.apply(weights_init)

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
        self.b = nn.Linear(input_dim, 1, bias=False)
        # self.apply(weights_init)

    def forward(self, x):
        en = self.main(x).squeeze(-1)
        return T.sum(x * x, 1) - self.b(x).squeeze() + en


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim + z_dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(.2),
            nn.Linear(dim, 1)
        )
        # self.apply(weights_init)

    def forward(self, x, z):
        x = torch.cat([x, z], -1)
        return self.main(x).squeeze(-1)
