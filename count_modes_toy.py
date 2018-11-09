import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

import torch

from utils import save_samples_energies
from data.toy import inf_train_gen
from networks.toy import Generator, EnergyModel, StatisticsNetwork
from train_functions import train_generator, train_energy_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

itr = inf_train_gen(args.dataset, args.batch_size)
netG = Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = EnergyModel(args.input_dim, args.dim).cuda()

#################################################
# Create Directories
#################################################
if root.exists():
    print('Loading model')
    netG.load_state_dict(
        torch.load(root / 'models/netG.pt')
    )
    # netE.load_state_dict(
    #     torch.load(root / 'models/netE.pt')
    # )
#################################################

thetas = np.arange(8) * (np.pi / 4)
radii = [2, 3, 4, 5]

modes = []
for r in radii:
    for t in thetas:
        theta = t + (r % 2) * (np.pi / 8)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        modes.append((x, y))
modes = np.asarray(modes, dtype='float32')

n_modes = []
ratio = []
for i in tqdm(range(args.iters)):
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x = netG(z).detach().cpu().numpy()

    dists = (x[:, :, None] - modes.T[None]) ** 2
    dists = dists.sum(1)

    realistic = dists < 0.02
    n_modes += [np.sign(realistic.sum(0)).sum()]
    ratio += [realistic.sum(-1).mean()]

print("No. of modes modeled = ", np.mean(n_modes))
print("Realistic ratio = ", np.mean(ratio))
