from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

import torch

from sampler import MALA_sampler
from networks.toy import Generator, EnergyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=.01)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

netG = Generator(args.z_dim, args.dim).cuda()
netE = EnergyModel(args.dim).cuda()

netG.eval()
netE.eval()

netG.load_state_dict(torch.load(
    root / 'models/netG.pt'
))
netE.load_state_dict(torch.load(
    root / 'models/netE.pt'
))


images = []
z = MALA_sampler(netG, netE, args)
x = netG(z).detach().cpu().numpy()



images = np.concatenate(images, 0)