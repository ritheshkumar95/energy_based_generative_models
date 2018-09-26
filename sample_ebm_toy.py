from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import os

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
images = []
z = MALA_sampler(netG, netE, args)
save_samples_energies(netG, netE, args, z=z)
