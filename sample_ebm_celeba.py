from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from sampler import MALA_corrected_sampler
# from inception_score import get_inception_score
from data.celeba import inf_train_gen
from networks.celeba import Generator, EnergyModel
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--mcmc_iters', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=.1)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_samples', type=int, default=5000)
    args = parser.parse_args()
    return args


args = parse_args()
root = Path(args.load_path)

itr = inf_train_gen(args.batch_size)
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
z = None
for i in tqdm(range(args.n_iters)):
    z, ratio = MALA_corrected_sampler(netG, netE, args, z=z)
    x_fake = netG(z).detach().cpu()
    save_image(x_fake, root / 'samples.png', normalize=True, nrow=4)
    images.append(imread(root / 'samples.png'))

print('Creating GIF....')
mimwrite('mcmc.gif', images)
