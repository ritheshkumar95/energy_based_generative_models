import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class KDEstimator(object):
    def __init__(self, x_real):
        self.x_real = x_real
        self.kde = KernelDensity(bandwidth=.5)
        self.kde.fit(x_real)
        self.log_p = self.kde.score_samples(x_real)

    def forward(self, netE, n_points=100):
        x = torch.from_numpy(self.x_real).cuda()
        e_x = -netE(x).detach().cpu().numpy()

        betas = []
        for i in range(x.size(0)):
            dists = (x[i:i+1] - x).norm(2, dim=1)
            j = random.choice(
                (-dists).topk(5)[1][1:]
            ).item()
            lhs = self.log_p[i] - self.log_p[j]
            rhs = e_x[i] - e_x[j]
            if np.sign(lhs / rhs) == 1:
                betas.append(lhs / rhs)

        return np.mean(betas), np.std(betas)


def learn_temperature(x_real, netE):
    beta = nn.Parameter(torch.ones(1).cuda())
    optimizer = torch.optim.LBFGS([beta], lr=0.8)

    def closure():
        optimizer.zero_grad()
        x = torch.from_numpy(x_real).cuda()
        median = x[:, 0].median()
        patch1 = x[x[:, 0] <= median]
        patch2 = x[x[:, 0] > median]

        vol_A = len(patch1) / (4 * (median + 2))
        vol_B = len(patch2) / (4 * (2 - median))

        lhs = torch.log(vol_A) - torch.log(vol_B)
        e_patch1 = -netE(patch1) * beta.abs()
        e_patch2 = -netE(patch2) * beta.abs()

        rhs = torch.logsumexp(e_patch1, 0) - torch.logsumexp(e_patch2, 0)
        loss = (lhs - rhs) ** 2

        loss.backward()
        return loss

    for i in range(5):
        optimizer.step(closure)

    return beta.item()


def save_samples(netG, args, z=None):
    if z is None:
        z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()

    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_fake[:, 0], x_fake[:, 1])
    return fig


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
