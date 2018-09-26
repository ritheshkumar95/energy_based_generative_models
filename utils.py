import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image


def log_sum_exp(vec):
    max_val = vec.max()[0]
    return max_val + (vec - max_val).exp().sum().log()


def save_samples_energies(netG, netE, args, n_points=500, z=None):
    root = Path(args.save_path) / 'images'

    if z is None:
        z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach()

    log_Z = log_sum_exp(
        -netE(x_fake).squeeze()
    ).item()

    plt.clf()
    x_fake = x_fake.cpu().numpy()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig(root / 'samples.png')

    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()

    energies = netE(grid).detach().cpu().numpy()
    e_grid = energies.reshape((n_points, n_points))
    p_grid = np.exp(- e_grid - log_Z)

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.colorbar()
    plt.savefig(root / 'energies.png')

    plt.clf()
    plt.imshow(p_grid, origin='lower')
    plt.colorbar()
    plt.savefig(root / 'densities.png')


def sample_images(netG, args):
    netG.eval()
    z = torch.randn(64, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()[:, :3]

    root = Path(args.save_path) / 'images'
    save_image(x_fake, root / 'samples.png', normalize=True)
    netG.train()
