import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_toy_samples(netG, args, z=None):
    if z is None:
        z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()

    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_fake[:, 0], x_fake[:, 1])
    return fig


def save_samples(netG, args):
    netG.eval()
    z = torch.randn(64, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()[:, :3]
    img = make_grid(x_fake, normalize=True)
    netG.train()
    return img


def save_energies(netE, args, n_points=500, beta=1.):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    with torch.no_grad():
        grid = torch.from_numpy(grid).float().cuda()
        e_grid = netE(grid) * beta

    p_grid = F.log_softmax(-e_grid, 0).exp()
    e_grid = e_grid.cpu().numpy().reshape((n_points, n_points))
    p_grid = p_grid.cpu().numpy().reshape((n_points, n_points))

    fig1 = plt.Figure()
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(e_grid, origin='lower')
    fig1.colorbar(im)

    plt.clf()
    fig2 = plt.Figure()
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(p_grid, origin='lower')
    fig2.colorbar(im)

    return fig1, fig2
