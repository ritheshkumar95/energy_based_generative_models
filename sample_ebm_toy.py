from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from imageio import imread, mimwrite

import torch

from utils import save_samples_energies
from sampler import MALA_sampler
from networks.toy import Generator, EnergyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--mcmc_iters', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=10)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

netG = Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = EnergyModel(args.input_dim, args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load(
    root / 'models/netG.pt'
))
netE.load_state_dict(torch.load(
    root / 'models/netE.pt'
))

args.save_path = Path('mcmc')
os.system('mkdir -p %s' % (args.save_path / 'images'))


root = Path(args.save_path) / 'images'
images = []
list_z = MALA_sampler(netG, netE, args)
list_img = []

x_vals = np.arange(-2, 3)
y_vals = np.arange(-2, 3)
modes = np.asarray(np.meshgrid(x_vals, y_vals))
modes = modes.reshape((2, 25)).T * (2 / 2.828)

for i, z in tqdm(enumerate(list_z)):
    x_fake = netG(z).detach()

    plt.clf()
    x_fake = x_fake.cpu().numpy()
    plt.scatter(x_fake[:, 0], x_fake[:, 1], s=20, alpha=.5)

    if args.dataset == '25gaussians':
        for x in range(-2, 3):
            for y in range(-2, 3):
                plt.scatter(modes[:, 0], modes[:, 1], c='red', alpha=.5, s=10)

    plt.title("Iter %d" % i)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig(root / 'samples.png')

    images.append(imread(root / 'samples.png'))

print('Creating GIF....')
mimwrite('mcmc.gif', images)
