import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, z_dim=128, dim=512):
        super(Generator, self).__init__()
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

    def forward(self, z):
        x = self.expand(z).view(z.size(0), -1, 2, 2)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, dim=512):
        super(Discriminator, self).__init__()
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
        return self.expand(out)


class Classifier(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super(Classifier, self).__init__()
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
        return self.classify(out)
